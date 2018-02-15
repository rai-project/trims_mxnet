#include "fmt/format.h"
#include "ipc.h"

#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <dmlc/omp.h>
#include <dmlc/recordio.h>
#include <dmlc/type_traits.h>
#include <nnvm/node.h>

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

#include <cuda_runtime_api.h>

using namespace upr;
using namespace mxnet;
using namespace std::string_literals;
using namespace grpc;

static const auto element_size = sizeof(float);

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
  std::map<std::string /* id::name */, cudaIpcMemHandle_t> open_handles{};

  std::string get_ipc_id(const std::string &id, const std::string &layer_name) {
    auto name = layer_name;

    static const std::string arg_prefix("arg:");
    if (string_starts_with(name, arg_prefix)) {
      name.erase(0, arg_prefix.size());
    }
    static const std::string aux_prefix("aux:");
    if (string_starts_with(name, aux_prefix)) {
      name.erase(0, aux_prefix.size());
    }

    const auto ipc_id = fmt::format("{}_{}", id, name);

    return ipc_id;
  }

  void make_ipc_handle(Layer *layer, const std::string &id,
                       const std::string &name, const float *data) {
    const auto ipc_id = get_ipc_id(id, name);

    cudaIpcMemHandle_t handle;

    CUDA_CHECK_CALL(cudaIpcGetMemHandle(&handle, device_ptr),
                    "failed to create a handle ref");

    open_handles.insert({ipc_id, handle});
    layer->set_ipc_handle((void *)&handle, sizeof(handle));
  }

  void make_ipc_handle(Layer *layer, const std::string &id,
                       const std::string &name, const NDArray &array) {
    const auto blob = array.data();
    auto data = blob.dptr<float>();
    make_ipc_handle(layer, id, name, data);
  }

  void make_ipc_handle(Layer *layer, const NDArray &array) {
    make_ipc_handle(layer, layer->id(), layer->name(), array);
  }

  void close_ipc_handle(std::string id, std::string name) {
    const auto ipc_id = get_ipc_id(id, name);
    const auto it = open_handles.find(ipc_id);
    if (it == open_handles.end()) {
      LOG(INFO) << "the ipc with id = " << ipc_id << " was not found";
      return;
    }
    LOG(INFO) << "TODO:: the ipc with id = " << ipc_id << " needs to be closed";
    open_handles.erase(ipc_id);
    return;
  }

  void to_shape(Shape *res, TShape shape) {
    res->set_rank(shape.ndim());
    for (const auto dim : shape) {
      res->add_dim(dim);
    }
  }
  void to_layer(Layer *layer, std::string name, NDArray cpu_array,
                int64_t ref_count) {
    const auto ctx = Context::GPU();
    const auto id = sole::uuid4().str();

    const auto array = cpu_array.Copy(ctx);

    array.WaitToRead();

    const auto blob = array.data();
    const auto shape = layer->mutable_shape();
    layer->set_id(id);
    layer->set_name(name);

    to_shape(shape, array.shape());

    layer->set_byte_count(blob.Size() * element_size);
    make_ipc_handle(layer, array);
    layer->set_device_raw_ptr((int64_t)blob.dptr<float>());
    layer->set_ref_count(ref_count);
  }

  void load_ndarray(::google::protobuf::RepeatedPtrField<Layer> *layers,
                    const ModelRequest *request, int64_t ref_count) {

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
      directory_path = get_model_directory_path(model_name);
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

    size_t ii = 0;
    for (const auto array : arrays) {
      const auto layer_name = layer_names[ii++];
      auto layer = layers->Add();
      to_layer(layer, layer_name, array, ref_count);
    }
  }

  void from_owned_layer(Layer *layer, const Layer &owned, int64_t ref_count) {

    const auto id = sole::uuid4().str();
    const auto shape = layer->mutable_shape();

    layer->set_id(id);
    layer->set_name(owned.name());

    shape->set_rank(owned.shape().rank());
    for (const auto dim : owned.shape().dim()) {
      shape->add_dim(dim);
    }
    layer->set_byte_count(owned.byte_count());
    layer->set_ipc_handle(
        make_ipc_handle(id, owned.name(), (float *)owned.device_raw_ptr()));
    layer->set_device_raw_ptr(owned.device_raw_ptr());
    layer->set_ref_count(ref_count);
  }

  void from_owned_modelhandle(ModelHandle *handle, const ModelHandle &owned,
                              int64_t ref_count) {

    const auto uuid = sole::uuid4().str();
    handle->set_id(uuid);
    handle->set_model_id(owned.model_id());
    handle->set_byte_count(owned.byte_count());

    auto layers = handle->mutable_layer();

    for (const auto owned_layer : owned.layer()) {
      auto trgt_layer = layers->Add();
      from_owned_layer(trgt_layer, owned_layer, ref_count);
    }
  }

public:
  grpc::Status Open(grpc::ServerContext *context, const ModelRequest *request,
                    ModelHandle *reply) override {
    std::lock_guard<std::mutex> lock(db_mutex_);

    LOG(INFO) << "opening " << request->name();

    auto it = memory_db_.find(request->name());
    if (it == memory_db_.end()) {
      const auto uuid = sole::uuid4().str();

      Model model;
      model.set_id(uuid);
      model.set_name(request->name());
      model.set_ref_count(0);

      auto owned_model = model.mutable_owned_model();
      owned_model->set_id("owned-by-" + uuid);
      owned_model->set_model_id(model.id());
      owned_model->set_byte_count(0);
      load_ndarray(owned_model->mutable_layer(), request, /*ref_count=*/0);

      int64_t byte_count = 0;
      for (const auto it : owned_model->layer()) {
        byte_count += it.byte_count();
      }

      owned_model->set_byte_count(byte_count);

      memory_db_[request->name()] = std::make_unique<Model>(model);
    }

    // std::cout << "keys = " << keys(memory_db_) << "\n";

    it = memory_db_.find(request->name());
    CHECK(it != memory_db_.end()) << "expecting the model to be there";

    CHECK(it->second != nullptr) << "expecting a valid model";

    // now we need to use the owned array to create
    // new memory handles
    LOG(INFO) << "creating shared handle from owned memory";
    it->second->set_ref_count(it->second->ref_count() + 1);
    auto handle = it->second->mutable_shared_model()->Add();
    from_owned_modelhandle(handle, it->second->owned_model(),
                           it->second->ref_count());
    LOG(INFO) << "sending " << it->second->owned_model().layer().size()
              << " layers to client";

    LOG(INFO) << "finished satisfying open request";

    reply->CopyFrom(*handle);

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
      return grpc::Status(grpc::NOT_FOUND, "unable to find handle with name "s +
                                               request->name() +
                                               " during info request");
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
      return grpc::Status(grpc::NOT_FOUND, "unable to find handle with name "s +
                                               request->model_id() +
                                               " during close request");
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

int main(int argc, const char *argv[]) {
  dmlc::InitLogging(argv[0]);
  int version = 0;
  const auto err = MXGetVersion(&version);
  if (err) {
    std::cerr << "error :: " << err << " while getting mxnet version\n";
  }
  std::cout << "in upd. using mxnet version = " << version
            << " on address  = " << server::address << "\n";

  system(std::string("rm -fr "s + IPC_HANDLES_BASE_PATH).c_str());
  system(std::string("mkdir -p "s + IPC_HANDLES_BASE_PATH).c_str());

  RunServer();

  return 0;
}