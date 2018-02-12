#include "ipc.h"
#include "mxnet/c_api.h"
#include "mxnet/c_predict_api.h"
#include "upr.grpc.pb.h"
#include "upr.pb.h"
#include <grpc++/grpc++.h>
#include <grpc/support/log.h>
#include <iostream>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <thread>

#include <cuda_runtime_api.h>

using namespace upr;
using namespace mxnet;
using namespace std::string_literals;
using namespace grpc;

static const auto BASE_PATH = "/tmp/persistent"s;

FILE *makefifo(std::string name, void *data) {
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

  for (int i = 0; i < sizeof(handle); i++) {
    ret = fprintf(fp, "%c", handle_buffer[i]);
    if (ret != 1)
      printf("ret = %d\n", ret);
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
  public:
    grpc::Status Open(grpc::ServerContext *context, const ModelRequest *request,
                      ModelHandle *reply) override {
      std::lock_guard<std::mutex> lock(db_mutex_);

      auto it = memory_db_.find(request->model_name());
      float *data;
      if (it == memory_db_.end()) {
        unsigned int byte_count = (unsigned int)DATA_SIZE;
        cudaMalloc((void **)&data, byte_count);
        cudaCheckErrors("malloc fail");
        cudaMemset(data, 0, byte_count);
        cudaCheckErrors("memset fail");
        // no cuda free in shutdown handling yet
        
        Model model;
        model.set_id(0);
        model.set_name(request->model_name());

        memory_db_[request->model_name()] =
            std::make_unique<Model>(model);

        std::cout << "keys = " << keys(memory_db_) << "\n";

        it = memory_db_.find(request->model_name());
      }

      const auto path = BASE_PATH + "/handle_"s + nextSuffix();
      const auto fp =
          makefifo(path, data);
      it->second->add_file_ptrs((int64_t)fp);
      it->second->set_ref_count(it->second->ref_count() + 1);

      const auto paths = it->second->paths();
      reply->set_path(it->second->paths(paths.size() - 1));
      reply->set_byte_count(it->second->byte_count());

      return grpc::Status::OK;
    }

    grpc::Status Info(grpc::ServerContext *context, const ModelRequest *request,
                      Model *reply) override {
      std::lock_guard<std::mutex> lock(db_mutex_);

      auto it = memory_db_.find(request->model_name());
      if (it == memory_db_.end()) {
        std::cout << "failed to info request. cannot find " << request->model_name()
                  << " in cache. "
                  << " cache = " << keys(memory_db_) << " \n";
        return grpc::Status(grpc::NOT_FOUND,
                            "unable to find handle with name "s +
                                request->model_name() + " during info request");
      }

      reply->CopyFrom(*it->second);

      return grpc::Status::OK;
    }

    grpc::Status Close(grpc::ServerContext *context,
                       const ModelHandle *request,
                       Void *reply) override {
      std::lock_guard<std::mutex> lock(db_mutex_);

      auto it = memory_db_.find(request->model_name());
      if (it == memory_db_.end()) {
        std::cout << "failed to close request\n";
        return grpc::Status(grpc::NOT_FOUND,
                            "unable to find handle with name "s +
                                request->model_name() + " during close request");
      }

      const auto path = request->file_path();

      const auto ref_count = it->second->ref_count() - 1;
      it->second->set_ref_count(ref_count);

      if (ref_count == 0) {
        // cudaFree((void *)it->second->raw_ptr());
        memory_db_.erase(it);
      }
      std::cout << "receive close request\n";

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
  std::string server_address("0.0.0.0:50051");
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