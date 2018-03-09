#include "fmt/format.h"
#include "ipc.h"

#include <algorithm>
#include <csignal>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <dmlc/omp.h>
#include <dmlc/recordio.h>
#include <dmlc/type_traits.h>
#include <fstream>
#include <future>
#include <nnvm/node.h>

#include "mxnet/c_api.h"
#include "mxnet/c_predict_api.h"
#include "sole/sole.hpp"
#include "upr.grpc.pb.h"
#include "upr.pb.h"
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <grpc++/grpc++.h>
#include <grpc/support/log.h>
#include <iostream>

#include <google/protobuf/util/time_util.h>

#include <cuda_runtime_api.h>

using namespace upr;
using namespace mxnet;
using namespace std::string_literals;
using namespace grpc;
using namespace google::protobuf::util;

static const auto element_size = sizeof(float);

template <typename K, typename V>
std::vector<K> keys(const std::map<K, V> &m) {
  std::vector<K> res;
  std::transform(m.begin(), m.end(), std::back_inserter(res),
                 [](const typename std::map<K, V>::value_type &pair) { return pair.first; });
  return res;
}

class RegistryImpl final : public Registry::Service {
private:
  struct ModelDeleter {
    void operator()(Model *ptr) const {
      auto span =
          start_span("deleting_model"s, "destroy", span_props{{"model_id", ptr->id()}, {"model_name", ptr->name()}});
      defer(stop_span(span));

      std::cout << "Model ptr" << ptr->id() << "\n";
      for (auto layer : ptr->owned_model().layer()) {
        void *dptr = (void *) layer.device_raw_ptr();
        if (dptr != nullptr) {
          cudaFree(dptr);
        }
      }
    }
  };
  using memory_db_t = std::map<std::string, std::unique_ptr<Model, ModelDeleter>>;
  // std::map<std::string /* id::name */, cudaIpcMemHandle_t> open_handles{};

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

  void make_ipc_handle(Layer *layer, const std::string &id, const std::string &name, float *device_ptr) {
    const auto ipc_id = get_ipc_id(id, name);

    auto span = start_span("ipc_get_memhandle"s, "ipc", span_props{{"id", id}, {"name", name}});
    defer(stop_span(span));

    cudaIpcMemHandle_t handle;
    CUDA_CHECK_CALL(cudaIpcGetMemHandle(&handle, (void *) device_ptr), "failed to create a handle ref");

    layer->set_ipc_handle(handle.reserved, CUDA_IPC_HANDLE_SIZE);
    // LOG(INFO) << "setting ipc handle " <<
    // utils::base64_encode(layer->ipc_handle()) << " for layer " <<
    // layer->name()
    //           << " with device_ptr = " << device_ptr << " and handle = " <<
    //           handle;
  }

  void make_ipc_handle(Layer *layer, const std::string &id, const std::string &name, NDArray &array) {

    auto span = start_span("make_ipc_handle"s, "ipc", span_props{{"id", id}, {"name", name}});
    defer(stop_span(span));

    const auto blob = array.data();
    auto data       = blob.dptr<float>();
    make_ipc_handle(layer, id, name, data);
  }

  void make_ipc_handle(Layer *layer, NDArray &array) {
    make_ipc_handle(layer, layer->id(), layer->name(), array);
  }

  void to_shape(Shape *res, TShape shape) {
    res->set_rank(shape.ndim());
    for (const auto dim : shape) {
      res->add_dim(dim);
    }
  }
  void to_layer(Layer *layer, std::string name, NDArray cpu_array, int64_t ref_count) {
    auto span =
        start_span("to_layer"s, "convert", span_props{{"ref_count", std::to_string(ref_count)}, {"name", name}});
    defer(stop_span(span));

    // LOG(INFO) << "converting " << name << " ndarray to protobuf
    // representation with ref_count = " << ref_count;
    const auto ctx = get_ctx();
    const auto id  = sole::uuid4().str();

    auto array = cpu_array.Copy(ctx);
    TIME_IT(array.WaitToRead()); // TODO:: REVISIT THIS

    const auto blob  = array.data();
    const auto shape = layer->mutable_shape();
    layer->set_id(id);
    layer->set_name(name);

    to_shape(shape, array.shape());

    layer->set_byte_count(blob.Size() * element_size);
    if (ref_count == -1) { // special value for owned model
      layer->set_ipc_handle("[owned]");
    } else {
      make_ipc_handle(layer, array);
    }
    // LOG(INFO) << "setting device_ptr = " << (int64_t) blob.dptr<float>();
    layer->set_device_raw_ptr((int64_t) blob.dptr<float>());
    layer->set_ref_count(ref_count);
  }

  void load_ndarray(::google::protobuf::RepeatedPtrField<Layer> *layers, const ModelRequest *request,
                    int64_t ref_count) {

    auto directory_path   = request->directory_path();
    const auto model_name = request->name();

    auto span = start_span("load_ndarray"s, "load",
                           span_props{{"ref_count", std::to_string(ref_count)}, {"mode_name", model_name}});
    defer(stop_span(span));

    // LOG(INFO) << fmt::format("loading ndarray directory_path = {} and
    // model_name = {}", directory_path, model_name);
    if (directory_path == "" && model_name == "") {
      const auto msg = "either the filepath or the model name must be specified in the open request"s;
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
    if (directory_path == "" && model_name != "") {
      // we need to load the model from the map
      directory_path = get_model_directory_path(model_name);
      // LOG(INFO) << fmt::format("using {} as the base directory for the
      // model_name = {}", directory_path, model_name);
    }
    if (directory_path != "" && !directory_exists(directory_path)) {
      const auto msg = fmt::format("directory_path {} does not exist", directory_path);
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }

    const auto params_path = get_model_params_path(model_name);
    if (!file_exists(params_path)) {
      const auto msg = fmt::format("the parameter file was not found in {}", params_path);
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }

    const auto symbol_path = get_model_symbol_path(model_name);
    if (!file_exists(symbol_path)) {
      const auto msg = fmt::format("the symbol file was not found in {}", symbol_path);
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }

    auto stream_span = start_span("create_dmlc_stream"s, "load",
                                  span_props{{"ref_count", std::to_string(ref_count)}, {"mode_name", model_name}});

    dmlc::Stream *fi(dmlc::Stream::Create(params_path.c_str(), "r", true));
    if (fi == nullptr) {
      const auto msg = fmt::format("unable to create a stream for {}", params_path);
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
    stop_span(stream_span);

    // LOG(INFO) << fmt::format("performing an ndarray load with params={} and
    // symbol={} paths", params_path, symbol_path);

    std::vector<NDArray> arrays{};
    std::vector<std::string> layer_names{};
    NDArray::Load(fi, &arrays, &layer_names);

    // LOG(INFO) << "starting to convert " << arrays.size() << " ndarrays to
    // protobuf representation";

    auto layers_span = start_span("to_layers"s, "load",
                                  span_props{{"ref_count", std::to_string(ref_count)}, {"mode_name", model_name}});
    size_t ii        = 0;
    for (const auto array : arrays) {
      const auto layer_name = layer_names[ii++];
      auto layer            = layers->Add();
      to_layer(layer, layer_name, array, ref_count);
    }
    stop_span(layers_span);
  }

  void from_owned_layer(Layer *layer, const Layer &owned, int64_t ref_count) {

    const auto id    = sole::uuid4().str();
    const auto shape = layer->mutable_shape();

    auto span = start_span("from_owned_layer"s, "load",
                           span_props{{"ref_count", std::to_string(ref_count)}, {"name", layer->name()}});
    defer(stop_span(span));

    // LOG(INFO) << "loading from owned layer for layer " << owned.name() << "
    // with ref_count = " << ref_count;

    layer->set_id(id);
    layer->set_name(owned.name());

    shape->set_rank(owned.shape().rank());
    for (const auto dim : owned.shape().dim()) {
      shape->add_dim(dim);
    }
    layer->set_byte_count(owned.byte_count());
    // LOG(INFO) << "creating ipc handle using device_ptr = " <<
    // owned.device_raw_ptr();
    make_ipc_handle(layer, id, owned.name(), (float *) owned.device_raw_ptr());
    // LOG(INFO) << "created ipc handle using device_ptr = " <<
    // owned.device_raw_ptr();
    layer->set_device_raw_ptr(owned.device_raw_ptr());
    layer->set_ref_count(ref_count);
  }

  void from_owned_modelhandle(ModelHandle *handle, const ModelHandle &owned, int64_t ref_count) {

    const auto uuid = sole::uuid4().str();

    auto span = start_span("from_owned_modelhandle"s, "load",
                           span_props{{"ref_count", std::to_string(ref_count)}, {"model_id", owned.model_id()}});
    defer(stop_span(span));

    handle->set_id(uuid);
    handle->set_model_id(owned.model_id());
    handle->set_byte_count(owned.byte_count());

    // LOG(INFO) << "loading from owned model";

    auto layers = handle->mutable_layer();

    for (const auto owned_layer : owned.layer()) {
      auto trgt_layer = layers->Add();
      from_owned_layer(trgt_layer, owned_layer, ref_count);
    }
  }

  size_t estimate_model_size(const ModelRequest *request) {
    static const auto estimation_rate = UPRD_ESTIMATION_RATE;
    const auto model_name             = request->name();
    const auto params_path            = get_model_params_path(model_name);

    auto span =
        start_span("estimate_model_size"s, "load",
                   span_props{{"estimation_rate", std::to_string(estimation_rate)}, {"model_name", request->name()}});
    defer(stop_span(span));

    std::ifstream in(params_path, std::ifstream::ate | std::ifstream::binary);
    return in.tellg() * estimation_rate;
  }

  bool perform_no_eviction(const ModelRequest *request, const size_t memory_size_request, const size_t memory_to_free) {
    return false;
  }

  bool perform_lru_eviction(const ModelRequest *request, const size_t memory_size_request,
                            const size_t memory_to_free) {
    size_t memory_freed = 0;
    bool ret            = false;

    while (memory_freed < memory_to_free) {
      if (memory_db_.empty()) {
        break;
      }
      auto min_element = std::min_element(memory_db_.begin(), memory_db_.end(),
                                          [](const memory_db_t::value_type &s1, const memory_db_t::value_type &s2) {
                                            if (s1.second->ref_count() > 0) {
                                              return false;
                                            }
                                            return s1.second->lru_timestamp() < s2.second->lru_timestamp();
                                          });
      if (min_element == memory_db_.end()) {
        break;
      }
      const auto byte_count = min_element->second->owned_model().byte_count();
      memory_usage_ -= byte_count;
      memory_freed += byte_count;
      memory_db_.erase(min_element);
    }

    if (memory_freed >= memory_to_free) {
      ret = true;
    }

    return ret;
  }

  bool perform_fifo_eviction(const ModelRequest *request,
                             const size_t memory_size_request,
                             const size_t memory_to_free) {
    size_t memory_freed = 0;
    bool ret            = false;

    while (memory_freed < memory_to_free) {
      if (memory_db_.empty()) {
        break;
      }
      auto min_element = std::min_element(memory_db_.begin(), memory_db_.end(),
                                          [](const memory_db_t::value_type &s1, const memory_db_t::value_type &s2) {
                                            if (s1.second->ref_count() > 0) {
                                              return false;
                                            }
                                            return s1.second->fifo_order() < s2.second->fifo_order();
                                          });
      if (min_element == memory_db_.end()) {
        break;
      }
      const auto byte_count = min_element->second->owned_model().byte_count();
      memory_usage_ -= byte_count;
      memory_freed += byte_count;
      memory_db_.erase(min_element);
    }

    if (memory_freed >= memory_to_free) {
      ret = true;
    }

    return ret;
  }

  bool perform_flush_eviction(const ModelRequest *request,
                              const size_t memory_size_request,
                              const size_t memory_to_free) {
    size_t memory_freed = 0;
    for (auto &&elem : memory_db_) {
      const auto byte_count = elem.second->owned_model().byte_count();
      memory_usage_ -= byte_count;
      memory_freed += byte_count;
    }
    memory_db_.clear();

    return memory_freed >= memory_to_free;
  }

  bool perform_lcu_eviction(const ModelRequest *request, const size_t memory_size_request,
                            const size_t memory_to_free) {
    size_t memory_freed = 0;
    bool ret            = false;

    while (memory_freed < memory_to_free) {
      if (memory_db_.empty()) {
        break;
      }
      auto min_element = std::min_element(memory_db_.begin(), memory_db_.end(),
                                          [](const memory_db_t::value_type &s1, const memory_db_t::value_type &s2) {
                                            if (s1.second->ref_count() > 0) {
                                              return false;
                                            }
                                            return s1.second->use_history().size() < s2.second->use_history().size();
                                          });
      if (min_element == memory_db_.end()) {
        break;
      }
      const auto byte_count = min_element->second->owned_model().byte_count();
      memory_usage_ -= byte_count;
      memory_freed += byte_count;
      memory_db_.erase(min_element);
    }

    if (memory_freed >= memory_to_free) {
      ret = true;
    }

    return ret;
  }

  // A eviction few strategies
  // - NEVER
  // - LRU
  // - FIFO
  // - RANDOM
  // - NEVER
  // - LCU -- least commnly used
  // - EAGER
  // - ALL
  bool perform_eviction(const ModelRequest *request, const size_t estimated_model_size, const size_t memory_to_free) {
    static const auto eviction_policy = UPRD_EVICTION_POLICY;

    auto span = start_span("perform_eviction"s, "load",
                           span_props{{"policy", eviction_policy},
                                      {"model_name", request->name()},
                                      {"estimated_model_size", std::to_string(estimated_model_size)},
                                      {"memory_to_free", std::to_string(memory_to_free)}});
    defer(stop_span(span));

    if (eviction_policy == "never") {
      return perform_no_eviction(request, estimated_model_size, memory_to_free);
    }
    if (eviction_policy == "lru") {
      return perform_lru_eviction(request, estimated_model_size, memory_to_free);
    }
    if (eviction_policy == "fifo") {
      return perform_fifo_eviction(request, estimated_model_size, memory_to_free);
    }
    if (eviction_policy == "flush" || eviction_policy == "all") {
      return perform_flush_eviction(request, estimated_model_size, memory_to_free);
    }
    if (eviction_policy == "lcu") { // least commnly used
      return perform_lcu_eviction(request, estimated_model_size, memory_to_free);
    }

    const auto msg = fmt::format("the eviction policy {} is not valid", eviction_policy);
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }

  void evict_if_needed(const ModelRequest *request) {
    static const auto max_memory_to_use = UPRD_MEMORY_PERCENTAGE * memory_total();
    const auto estimated_model_size     = estimate_model_size(request);
    if (estimated_model_size > max_memory_to_use) {
      const auto msg = fmt::format("cannot allocate memory. requsting an estimated {} bytes "
                                   "of memory, while only {} is allocated to be used",
                                   estimated_model_size, max_memory_to_use);
      throw std::runtime_error(msg);
    }
    if (memory_usage_ + estimated_model_size > max_memory_to_use) {
      const auto memory_to_free = estimated_model_size + memory_usage_ - max_memory_to_use;
      const auto ok             = perform_eviction(request, estimated_model_size, memory_to_free);
      if (!ok) {
        static const auto eviction_policy = UPRD_EVICTION_POLICY;
        const auto msg                    = fmt::format("cannot fulfill memory allocation. requesting an estimated {} "
                                     "bytes of memory to be freed, while only {} is allocated to be "
                                     "used using the {} eviction strategy",
                                     memory_to_free, max_memory_to_use, eviction_policy);
        throw std::runtime_error(msg);
      }
      return;
    }
    return;
  }

  size_t fifo_order{0};

public:
  // control memory usage by percentage of gpu
  grpc::Status Open(grpc::ServerContext *context, const ModelRequest *request, ModelHandle *reply) override {
    std::lock_guard<std::mutex> lock(db_mutex_);

    auto span = start_span("open"s, "grpc", span_props{{"model_name", request->name()}});
    defer(stop_span(span));

    // LOG(INFO) << "opening " << request->name();

    auto it = memory_db_.find(request->name());
    if (it == memory_db_.end()) {
      const auto uuid = sole::uuid4().str();

      auto owned_span = start_span("load_owned"s, "load", span_props{{"model_name", request->name()}});
      defer(stop_span(owned_span));

      try {
        evict_if_needed(request);
      } catch (const std::runtime_error &error) {
        return grpc::Status(grpc::RESOURCE_EXHAUSTED, error.what());
      }

      Model *model = new Model();
      model->set_id(uuid);
      model->set_name(request->name());
      model->set_ref_count(0);

      auto owned_model = model->mutable_owned_model();
      owned_model->set_id("owned-by-" + uuid);
      owned_model->set_model_id(model->id());
      owned_model->set_byte_count(0);

      load_ndarray(owned_model->mutable_layer(), request, /*ref_count=*/-1);

      int64_t byte_count = 0;
      for (const auto it : owned_model->layer()) {
        byte_count += it.byte_count();
      }

      owned_model->set_byte_count(byte_count);

      model->set_fifo_order(fifo_order++);

      memory_db_[request->name()] = std::unique_ptr<Model, ModelDeleter>(model);
      memory_usage_ += byte_count;
    }

    auto shared_span = start_span("make_shared"s, "share", span_props{{"model_name", request->name()}});
    defer(stop_span(shared_span));

    // LOG(INFO) << "done with creating owned model";

    it = memory_db_.find(request->name());
    CHECK(it != memory_db_.end()) << "expecting the model to be there";

    CHECK(it->second != nullptr) << "expecting a valid model";

    // now we need to use the owned array to create
    // new memory handles
    // LOG(INFO) << "creating shared handle from owned memory";
    it->second->set_ref_count(it->second->ref_count() + 1);
    auto handle = it->second->mutable_shared_model()->Add();
    from_owned_modelhandle(handle, it->second->owned_model(), it->second->ref_count());
    // LOG(INFO) << "sending " << it->second->owned_model().layer().size() << "
    // layers to client";

    // LOG(INFO) << "finished satisfying open request";

    auto h = it->second->mutable_use_history()->Add();
    h->CopyFrom(TimeUtil::GetCurrentTime());

    auto t = it->second->mutable_lru_timestamp();
    t->CopyFrom(TimeUtil::GetCurrentTime());

    reply->CopyFrom(*handle);

    return grpc::Status::OK;
  }

  grpc::Status Info(grpc::ServerContext *context, const ModelRequest *request, Model *reply) override {
    std::lock_guard<std::mutex> lock(db_mutex_);

    auto span = start_span("info"s, "grpc", span_props{{"model_name", request->name()}});
    defer(stop_span(span));

    auto it = memory_db_.find(request->name());
    if (it == memory_db_.end()) {
      LOG(ERROR) << "failed to info request. cannot find " << request->name() << " in cache. "
                 << " cache = " << keys(memory_db_) << " \n";
      return grpc::Status(grpc::NOT_FOUND,
                          "unable to find handle with name "s + request->name() + " during info request");
    }

    reply->CopyFrom(*it->second);

    return grpc::Status::OK;
  }

  std::string find_model_name_by_model_id(std::string model_id) {
    const auto loc = std::find_if(memory_db_.begin(), memory_db_.end(),
                                  [&model_id](const memory_db_t::value_type &k) { return k.second->id() == model_id; });
    if (loc == memory_db_.end()) {
      return "";
    }
    return loc->second->name();
  }

  void destroy_model_handle(const ModelHandle &handle) {
  }

  grpc::Status Close(grpc::ServerContext *context, const ModelHandle *request, Void *reply) override {
    std::lock_guard<std::mutex> lock(db_mutex_);

    auto span = start_span("close"s, "grpc", span_props{{"id", request->id()}, {"model_id", request->model_id()}});
    defer(stop_span(span));

    const auto model_name = find_model_name_by_model_id(request->model_id());
    if (model_name == "") {
      LOG(ERROR) << "failed to close request\n";
      return grpc::Status(grpc::NOT_FOUND,
                          "unable to find model name with id "s + request->model_id() + " during close request");
    }
    auto model_entry = memory_db_.find(model_name);
    if (model_entry == memory_db_.end()) {
      LOG(ERROR) << "failed to close request\n";
      return grpc::Status(grpc::NOT_FOUND, "unable to find handle with name "s + model_name + " during close request");
    }

    const auto handle_id    = request->id();
    const auto shared_model = model_entry->second->mutable_shared_model();
    for (auto it = shared_model->begin(); it != shared_model->end(); it++) {
      auto model = *it;
      if (model.id() != handle_id) {
        continue;
      }
      destroy_model_handle(model);
      model_entry->second->mutable_shared_model()->erase(it);
      break;
    }

    const auto ref_count = model_entry->second->ref_count() - 1;
    model_entry->second->set_ref_count(ref_count);

    if (ref_count == 0) {
      static const auto eviction_policy = UPRD_EVICTION_POLICY;
      if (eviction_policy == "eager") {
        const auto byte_count = model_entry->second->owned_model().byte_count();
        memory_usage_ -= byte_count;
        model_entry->second.reset(nullptr);
        memory_db_.erase(model_entry);
      }
    }

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

  memory_db_t memory_db_;
  int64_t memory_usage_{0};

  // Mutex serializing access to the map.
  std::mutex db_mutex_;
};

std::promise<void> exit_requested;

int main(int argc, const char *argv[]) {
  static const auto eviction_policy   = UPRD_EVICTION_POLICY;
  static const auto estimation_rate   = UPRD_ESTIMATION_RATE;
  static const auto max_memory_to_use = UPRD_MEMORY_PERCENTAGE * memory_total();
  int version                         = 0;
  const auto err                      = MXGetVersion(&version);
  if (err) {
    std::cerr << "error :: " << err << " while getting mxnet version\n";
  }

  const std::string profile_default_path{"server_profile.json"};
  const auto profile_path = dmlc::GetEnv("UPR_PROFILE_TARGET", profile_default_path);
  LOG(INFO) << "in uprd. using mxnet version = " << version << " running on address  = " << server::address << "\n";
  LOG(INFO) << "eviction_policy = " << eviction_policy << "\n"
            << "estimation_rate = " << estimation_rate << "\n"
            << "max_memory_to_use = " << max_memory_to_use << "\n"
            << "profile_path = " << profile_path << "\n";

  force_runtime_initialization();

  MXPredInit();

  MXSetProfilerConfig(1, profile_default_path.c_str());

  // Start profiling
  MXSetProfilerState(1);

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
  auto serveFn = [&]() { server->Wait(); };

  std::thread serving_thread(serveFn);

  auto signal_handler = [](int s) { exit_requested.set_value(); };
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  std::signal(SIGQUIT, signal_handler);

  auto f = exit_requested.get_future();
  f.wait();

  MXSetProfilerState(0);

  server->Shutdown();
  serving_thread.join();

  return 0;
}
