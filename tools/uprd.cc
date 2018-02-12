#include "fmt/format.h"
#include "ipc.h"
#include "mxnet/c_api.h"
#include "mxnet/c_predict_api.h"
#include "prettyprint.hpp"
#include "sole/sole.hpp"
#include "upr.grpc.pb.h"
#include "upr.pb.h"
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <grpc++/grpc++.h>
#include <grpc/support/log.h>
#include <iostream>

#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>

using namespace upr;
using namespace mxnet;
using namespace std::string_literals;
using namespace grpc;

static const auto BASE_PATH = "/tmp/persistent"s;
static const auto CARML_HOME_BASE_DIR = "/home/abduld/carml/data/mxnet/"s;

static std::map<std::string, std::string> model_directory_paths{
    {"alexnet", CARML_HOME_BASE_DIR + "alexnet"},
    {"vgg16", CARML_HOME_BASE_DIR + "vgg16"}};

#define CUDA_CHECK_CALL(func, msg)                                             \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                   \
        << "CUDA[" << msg << "]:: " << cudaGetErrorString(e);                  \
  }

static bool directory_exists(const std::string &path) {
  struct stat sb;
  if (stat(path.c_str(), &sb) == -1) {
    int errsv = errno;
    return false;
  }

  if ((sb.st_mode & S_IFMT) == S_IFDIR) {
    return true;
  }
  return false;
}

static bool file_exists(const std::string &path) {

  struct stat sb;
  if (stat(path.c_str(), &sb) == -1) {
    int errsv = errno;
    return false;
  }

  if ((sb.st_mode & S_IFMT) == S_IFDIR) {
    return false;
  }
  return true;
}

static FILE *makefifo(std::string name, void *data) {
  cudaIpcMemHandle_t handle;
  cudaIpcGetMemHandle(&handle, data);

  const auto cmd = std::string("rm -f ") + name;
  system(cmd.c_str());                  // remove any debris
  int ret = mkfifo(name.c_str(), 0600); // create fifo
  if (ret != 0) {
    printf("mkfifo error: %d\n", ret);
  }

  system(std::string("rm -f "s + name).c_str());

  auto fp = fopen(name.c_str(), "w");
  if (fp == NULL) {
    printf("fifo open fail \n");
    return nullptr;
  }

  auto fd = fileno(fp);
  const auto flags = fcntl(fd, F_GETFL);
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);

  std::cout << "in process mkfifo"
            << "\n";

  unsigned char handle_buffer[sizeof(handle) + 1];
  memset(handle_buffer, 0, sizeof(handle) + 1);
  memcpy(handle_buffer, (unsigned char *)(&handle), sizeof(handle));

  for (size_t ii = 0; ii < sizeof(handle); ii++) {
    ret = fprintf(fp, "%c", handle_buffer[ii]);
    if (ret != 1) {
      printf("ret = %d\n", ret);
      return nullptr;
    }
  }

  fclose(fp);

  std::cout << "wrote " << sizeof(handle) << " bytes to " << name << "\n";

  return fp;
}

template <typename K, typename V> std::vector<K> keys(const std::map<K, V> &m) {
  std::vector<K> res;
  std::transform(m.begin(), m.end(), std::back_inserter(res),
                 [](const typename std::map<K, V>::value_type &pair) {
                   return pair.first;
                 });
  return res;
}

class RegistryImpl final : public Registry::Service {
private:
  void load_ndarray(::google::protobuf::RepeatedPtrField<Layer> *layers,
                    const ModelRequest *request) {

    auto directory_path = request->directory_path();
    const auto model_name = request->name();

    LOG(INFO) << fmt::format(
        "loading ndarray directory_path = {} and model_name = {}",
        directory_path, model_name);
    if (directory_path == "" && model_name == "") {
      const auto msg =
          "either the filepath or the model name must be specified in the open request"s;
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
    if (directory_path == "" && model_name != "") {
      // we need to load the model from the map
      const auto pth = model_directory_paths.find(model_name);
      if (pth == model_directory_paths.end()) {
        const auto msg = fmt::format(
            "the model path for {} was not found in the model directory",
            model_name);
        LOG(ERROR) << msg;
        throw std::runtime_error(msg);
      }
      directory_path = pth->second;
      LOG(INFO) << fmt::format(
          "using {} as the base directory for the model_name = {}",
          directory_path, model_name);
    }
    if (directory_path != "" && !directory_exists(directory_path)) {
      const auto msg =
          fmt::format("directory_path {} does not exist", directory_path);
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }

    const auto params_path = directory_path + "/model.params";

    if (!file_exists(params_path)) {
      const auto msg =
          fmt::format("the parameter file was not found in {}", params_path);
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }

    const auto symbol_path = directory_path + "/model.symbol";
    if (!file_exists(symbol_path)) {
      const auto msg =
          fmt::format("the symbol file was not found in {}", symbol_path);
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }

    dmlc::Stream *fi(dmlc::Stream::Create(params_path.c_str(), "r", true));
    if (fi == nullptr) {
      const auto msg =
          fmt::format("unable to create a stream for {}", params_path);
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }

    std::vector<NDArray> arrays{};
    std::vector<std::string> layer_names{};
    NDArray::Load(fi, &arrays, &layer_names);

    for (const auto array : arrays) {
      LOG(ERROR) << array.shape();
    }
  }

public:
  grpc::Status Open(grpc::ServerContext *context, const ModelRequest *request,
                    ModelHandle *reply) override {
    std::lock_guard<std::mutex> lock(db_mutex_);

    auto it = memory_db_.find(request->name());
    float *data;
    if (it == memory_db_.end()) {

      const auto uuid = sole::uuid1().pretty();

      Model model;
      model.set_id(uuid);
      model.set_name(request->name());

      memory_db_[request->name()] = std::make_unique<Model>(model);

      // std::cout << "keys = " << keys(memory_db_) << "\n";

      it = memory_db_.find(request->name());

      ModelHandle owned_model;
      owned_model.set_id("owned-by-" + uuid);
      owned_model.set_model_id(model.id());
      owned_model.set_byte_count(0);
      load_ndarray(owned_model.mutable_layers(), request);

      int64_t byte_count = 0;
      for (const auto it : owned_model.layers()) {
        byte_count += it.byte_count();
      }
      owned_model.set_byte_count(byte_count);
    }

    const auto path = BASE_PATH + "/handle_"s + nextSuffix();
#ifdef IMPLEMENT_ND_ARRAYLOAD
    const auto fp = makefifo(path, data);
    it->second->add_file_ptrs((int64_t)fp);
    it->second->set_ref_count(it->second->ref_count() + 1);

    const auto paths = it->second->paths();
    reply->set_path(it->second->paths(paths.size() - 1));
    reply->set_byte_count(it->second->byte_count());
#else
    std::cerr << "implement ndarray load\n";
#endif
    return grpc::Status::OK;
  }

  grpc::Status Info(grpc::ServerContext *context, const ModelRequest *request,
                    Model *reply) override {
    std::lock_guard<std::mutex> lock(db_mutex_);

    auto it = memory_db_.find(request->name());
    if (it == memory_db_.end()) {
      std::cout << "failed to info request. cannot find " << request->name()
                << " in cache. "
                << " cache = " << keys(memory_db_) << " \n";
      return grpc::Status(grpc::NOT_FOUND,
                          "unable to find handle with name "s +
                              request->name() + " during info request");
    }

    reply->CopyFrom(*it->second);

    return grpc::Status::OK;
  }

  grpc::Status Close(grpc::ServerContext *context, const ModelHandle *request,
                     Void *reply) override {
    std::lock_guard<std::mutex> lock(db_mutex_);

    auto it = memory_db_.find(request->model_id());
    if (it == memory_db_.end()) {
      std::cout << "failed to close request\n";
      return grpc::Status(grpc::NOT_FOUND,
                          "unable to find handle with name "s +
                              request->model_id() + " during close request");
    }

#ifdef REF_COUNT_ENABLED
    const auto path = request->file_path();

    const auto ref_count = it->second->ref_count() - 1;
    it->second->set_ref_count(ref_count);

    if (ref_count == 0) {
      // cudaFree((void *)it->second->raw_ptr());
      memory_db_.erase(it);
    }
    std::cout << "receive close request\n";
#else
    std::cerr << "TODO:: enable decremenet refcount on close\n";
#endif
    return grpc::Status::OK;
  }

private:
  std::string nextSuffix() {
    static std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_int_distribution<> dis(1, 1e9);
    return std::to_string(dis(gen));
  }

  // The actual database.
  std::map<std::string, std::unique_ptr<Model>> memory_db_;

  // Mutex serializing access to the map.
  std::mutex db_mutex_;
};

void RunServer() {
  cudaSetDevice(0);
  std::string server_address(server::address);
  RegistryImpl service;

  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main() {
  int version = 0;
  const auto err = MXGetVersion(&version);
  if (err) {
    std::cerr << "error :: " << err << " while getting mxnet version\n";
  }
  std::cout << "in upd. using mxnet version = " << version
            << " on address  = " << server::address << "\n";

  system(std::string("rm -fr "s + BASE_PATH).c_str());
  system(std::string("mkdir -p "s + BASE_PATH).c_str());

  RunServer();

  return 0;
}