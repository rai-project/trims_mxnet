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
#include <shared_mutex>

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

#include <hopscotch/hopscotch_map.h>
#include <hopscotch/hopscotch_sc_map.h>

using namespace upr;
using namespace mxnet;
using namespace grpc;
using namespace google::protobuf::util;

static const auto element_size = sizeof(float);

template <typename K, typename V>
std::vector<K> keys(const tsl::hopscotch_sc_map<K, V> &m) {
  std::vector<K> res;
  std::transform(m.begin(), m.end(), std::back_inserter(res),
                 [](const typename std::map<K, V>::value_type &pair) { return pair.first; });
  return res;
}

class RegistryImpl final : public Registry::Service {
private:
  struct model_info {
    SharingGranularity granularity;
    void *base_ptr{nullptr};
    size_t byte_count{0};
    std::vector<TShape> shapes{};
    std::vector<void *> data{};
    std::vector<size_t *> offsets{};
    std::vector<std::string> layer_names{};
  };
  using cpu_persistent_data_t = std::map<std::string, model_info *>;
  using memory_db_t           = tsl::hopscotch_sc_map<std::string, Model *, std::hash<std::string>>;

  cpu_persistent_data_t cpu_persistent_data{};

  void model_delete(Model *ptr) {
    if (ptr == nullptr) {
      return;
    }
    auto span =
        start_span("deleting_model", "destroy", span_props{{"model_id", ptr->id()}, {"model_name", ptr->name()}});
    defer(stop_span(span));

    LOG(INFO) << "deleting model id=" << ptr->id();
    auto owned = ptr->owned_model();
    if (owned.sharing_granularity() == SharingGranularity_Layer) {
      for (auto layer : owned.layer()) {
        void *dptr = (void *) layer.device_raw_ptr();
        if (dptr != nullptr) {
          cudaFree(dptr);
        }
      }
      return;
    }
    if (owned.sharing_granularity() == SharingGranularity_Model) {
      void *dptr = (void *) ptr->device_raw_ptr();
      if (dptr != nullptr) {
        cudaFree(dptr);
      }
      return;
    }
    throw std::runtime_error("invalid sharing granularity");
  }

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

  cudaIpcMemHandle_t make_ipc_handle(float *device_ptr) {
    cudaIpcMemHandle_t handle;
    CUDA_CHECK_CALL(cudaIpcGetMemHandle(&handle, (void *) device_ptr), "failed to create a handle ref");
    return handle;
  }

  void make_ipc_handle(Layer *layer, const std::string &id, const std::string &name, float *device_ptr) {
    const auto ipc_id = get_ipc_id(id, name);

    auto span = start_span("ipc_get_memhandle", "ipc", span_props{{"id", id}, {"name", name}});

    auto handle = make_ipc_handle(device_ptr);

    layer->set_ipc_handle(handle.reserved, CUDA_IPC_HANDLE_SIZE);

    stop_span(span);
  }

  void make_ipc_handle(Layer *layer, const std::string &id, const std::string &name, const NDArray &array) {

    auto span = start_span("make_ipc_handle", "ipc", span_props{{"id", id}, {"name", name}});
    defer(stop_span(span));

    const auto blob = array.data();
    auto data       = blob.dptr<float>();
    make_ipc_handle(layer, id, name, data);
  }

  void make_ipc_handle(Layer *layer, const NDArray &array) {
    make_ipc_handle(layer, layer->id(), layer->name(), array);
  }

  void to_shape(Shape *res, TShape shape) {
    res->set_rank(shape.ndim());
    for (const auto dim : shape) {
      res->add_dim(dim);
    }
  }

  void to_layer_from_disk(Layer *layer, const std::string &name, const NDArray &array, int64_t ref_count,
                          cudaStream_t stream = 0) {
    auto span = start_span("to_layer_from_disk", "convert",
                           span_props{{"ref_count", std::to_string(ref_count)}, {"name", name}});
    defer(stop_span(span));

    // LOG(INFO) << "converting " << name << " ndarray to protobuf
    // representation with ref_count = " << ref_count;
    const auto ctx = get_ctx();
    const auto id  = sole::uuid4().str();

    const auto blob       = array.data();
    const float *cpu_ptr  = (float *) blob.dptr_;
    const auto byte_count = blob.Size() * element_size;

    void *device_ptr;
    CUDA_CHECK_CALL(cudaMalloc(&device_ptr, byte_count), "cannot allocate layer");
    CUDA_CHECK_CALL(cudaMemcpyAsync(device_ptr, cpu_ptr, byte_count, cudaMemcpyHostToDevice, stream),
                    "cannot copy layer");

    const auto shape = layer->mutable_shape();
    layer->set_id(id);
    layer->set_name(name);

    to_shape(shape, array.shape());

    layer->set_byte_count(byte_count);
    if (ref_count == -1) { // special value for owned model
      layer->set_ipc_handle("[owned]");
    } else {
      make_ipc_handle(layer, array);
    }
    // LOG(INFO) << "setting device_ptr = " << (int64_t) blob.dptr<float>();
    layer->set_device_raw_ptr((int64_t) blob.dptr<float>());
    layer->set_sharing_granularity(SharingGranularity_Layer);
    layer->set_ref_count(ref_count);
  }

  void to_layer_from_cpu_mem(Layer *layer, std::string name, const void *ptr, const TShape &tshape, int64_t ref_count,
                             cudaStream_t stream = 0) {
    auto span = start_span("to_layer_from_cpu_mem", "convert",
                           span_props{{"ref_count", std::to_string(ref_count)}, {"name", name}});
    defer(stop_span(span));

    // LOG(INFO) << "converting " << name << " ndarray to protobuf
    // representation with ref_count = " << ref_count;
    const auto id = sole::uuid4().str();

    const size_t type_size  = element_size;
    const size_t byte_count = type_size * tshape.Size();

    float *dev_ptr = nullptr;

    CUDA_CHECK_CALL(cudaMalloc(&dev_ptr, byte_count), "failed to allocate device pointer while creating cpu memory");
    CUDA_CHECK_CALL(cudaMemcpyAsync(dev_ptr, cpu_ptr, byte_count, cudaMemcpyHostToDevice, stream),
                    "faile to copy cpu memory to gpu");

    const auto shape = layer->mutable_shape();
    layer->set_id(id);
    layer->set_name(name);

    to_shape(shape, tshape);

    layer->set_byte_count(byte_count);
    if (ref_count == -1) { // special value for owned model
      layer->set_ipc_handle("[owned]");
    } else {
      make_ipc_handle(layer, id, name, dev_ptr);
    }
    // LOG(INFO) << "setting device_ptr = " << (int64_t) blob.dptr<float>();
    layer->set_device_raw_ptr((int64_t) dev_ptr);
    layer->set_ref_count(ref_count);
  }

  model_info *to_model_info_for_model_sharing_granularity(const std::vector<NDArray> &arrays,
                                                          const std::vector<std::string> &layer_names) {
    size_t total_byte_count = 0;
    const size_t type_size  = element_size;

    for (const auto &array : arrays) {
      const auto blob       = array.data();
      const auto byte_count = type_size * blob.Size();
      total_byte_count += type_size * blob.Size();
    }

    void *base_ptr = malloc(total_byte_count);
    CHECK(base_ptr != NULL, "unable to allocate cpu memory");

    size_t ii     = 0;
    size_t offset = 0;

    auto info         = new model_info{};
    info->granularity = SharingGranularity_Model;
    info->base_ptr    = base_ptr;
    info->byte_count  = total_byte_count;

    for (const auto &array : arrays) {
      const auto blob       = array.data();
      const char *arry_ptr  = (char *) blob.dptr_;
      const auto byte_count = type_size * blob.Size();
      const auto layer_name = layer_names[ii++];

      auto cpu_mem = base_ptr + offset;
      memcpy(cpu_mem, arry_ptr, byte_count);

      info->shapes.emplace_back(array.shape());
      info->data.emplace_back(cpu_mem);
      info->layer_names.emplace_back(layer_name);
      info->offsets.emplace_back(offset);

      offset += byte_count;
    }

    return info;
  }

  void to_layers_from_model_info_for_model_granularity(::google::protobuf::RepeatedPtrField<Layer> *layers layers,
                                                       const auto model_info *info,
                                                       int64_t ref_count,
                                                       cudaStream_t stream = 0) {

    CHECK(info->granularity == SharingGranularity_Model, "expecting model granularity");
    if (info->granularity != SharingGranularity_Model) {
      throw std::runtime_error("expecting model granularity";)
    }

    void *device_ptr;
    CUDA_CHECK_CALL(cudaMalloc(&device_ptr, info->byte_count),
                    "cannot allocate to_layers_from_model_info_for_model_granularity");
    CUDA_CHECK_CALL(cudaMemcpyAsync(device_ptr, info->base_ptr, info->byte_count, cudaMemcpyHostToDevice, stream),
                    "cannot perform to_layers_from_model_info_for_model_granularity");

    auto ipc_handle = make_ipc_handle(device_ptr);

    for (size_t ii = 0; ii < info->layer_names.size(); ii++) {
      auto layer            = layers->Add();
      const auto layer_name = info->layer_names[ii];
      const auto tshape     = info->shapes[ii];
      const auto offset     = info->offset[ii];

      auto span = start_span("to_layer_from_cpu_mem", "convert",
                             span_props{{"ref_count", std::to_string(ref_count)},
                                        {"name", name},
                                        {"granularity", SharingGranularity_Name(SharingGranularity_Model)}});

      // LOG(INFO) << "converting " << name << " ndarray to protobuf
      // representation with ref_count = " << ref_count;
      const auto id = sole::uuid4().str();

      const auto shape = layer->mutable_shape();
      layer->set_id(id);
      layer->set_name(name);

      to_shape(shape, tshape);

      layer->set_byte_count(byte_count);
      if (ref_count == -1) { // special value for owned model
        layer->set_ipc_handle("[owned]");
      } else {
        layer->set_ipc_handle(ipc_handle);
      }
      // LOG(INFO) << "setting device_ptr = " << (int64_t) blob.dptr<float>();
      layer->set_sharing_granularity(SharingGranularity_Model);
      layer->set_offset(offset);
      layer->set_device_raw_ptr((int64_t) device_ptr);
      layer->set_ref_count(ref_count);

      stop_span(span);
    }
  }

  void load_from_cpu_mem(::google::protobuf::RepeatedPtrField<Layer> *layers, const std::string &model_name,
                         int64_t ref_count, cudaStream_t stream = 0) {
    auto e    = cpu_persistent_data.find(model_name);
    auto info = e->second;
    if (info->granularity == SharingGranularity_Layer) {
      for (size_t ii = 0; ii < info->layer_names.size(); ii++) {
        auto layer            = layers->Add();
        const auto layer_name = info->layer_names[ii];
        const auto cpu_ptr    = info->data[ii];
        const auto shape      = info->shapes[ii];
        to_layer_from_cpu_mem(layer, layer_name, cpu_ptr, shape, ref_count);
      }
      return;
    }
    if (info->granularity == SharingGranularity_Model) {
      auto info = to_model_info_for_model_sharing_granularity(arrays, layer_names);
      to_layers_from_model_info_for_model_granularity(layers, info, ref_count, stream);
      delete info;
      return;
    }

    throw std::runtime_error("invalid sharing granularity");
  }

  bool is_persistent_on_cpu(const std::string &model_name) {
    if (!UPRD_PERSIST_CPU) {
      return false;
    }
    return cpu_persistent_data.find(model_name) != cpu_persistent_data.end();
  }

  void persist_on_cpu(const std::string &model_name, const NDArray &array, const std::string &layer_name) {
    if (cpu_persistent_data.find(model_name) == cpu_persistent_data.end()) {
      auto model_info         = new model_info{};
      model_info->granularity = SharingGranularity_Layer;
      cpu_persistent_data.insert({model_name, model_info});
    }
    auto e    = cpu_persistent_data.find(model_name);
    auto info = e->second;

    const auto blob      = array.data();
    const char *arry_ptr = (char *) blob.dptr_;

    info->shapes.emplace_back(array.shape());
    info->data.emplace_back(arry_ptr);
    info->layer_names.emplace_back(layer_name);
  }

  void persist_on_cpu(const SharingGranularity &sharing_granularity, const std::string &model_name,
                      const std::vector<NDArray> &arrays, const std::vector<std::string> &layer_names) {
    if (sharing_granularity == SharingGranularity_Layer) {
      size_t ii = 0;
      for (const auto &array : arrays) {
        const auto layer_name = layer_names[ii++];
        persist_on_cpu(model_name, array, layer_name);
      }
      return;
    }
    if (sharing_granularity == SharingGranularity_Model) {
      auto info = to_model_info_with_model_sharing(arrays, layer_names);
      cpu_persistent_data.insert({model_name, info});
      return;
    }

    throw std::runtime_error("invalid sharing granularity");
  }

  void load_ndarray(::google::protobuf::RepeatedPtrField<Layer> *layers, const ModelRequest *request, int64_t ref_count,
                    cudaStream_t stream = 0) {

    const auto model_name = request->name();

    if (is_persistent_on_cpu(model_name)) {
      auto layers_span = start_span("to_layers_from_cpu_mem", "load",
                                    span_props{{"ref_count", std::to_string(ref_count)}, {"mode_name", model_name}});
      load_from_cpu_mem(layers, model_name, ref_count);
      stop_span(layers_span);
      return;
    }

    auto directory_path = request->directory_path();

    auto span = start_span("load_ndarray", "load",
                           span_props{{"ref_count", std::to_string(ref_count)}, {"mode_name", model_name}});
    defer(stop_span(span));

    // LOG(INFO) << fmt::format("loading ndarray directory_path = {} and
    // model_name = {}", directory_path, model_name);
    if (directory_path == "" && model_name == "") {
      const std::string msg = "either the filepath or the model name must be specified in the open request";
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

    // LOG(INFO) << fmt::format("performing an ndarray load with params={} and
    // symbol={} paths", params_path, symbol_path);

    std::vector<NDArray> arrays{};
    std::vector<std::string> layer_names{};
    auto stream_span = start_span("create_dmlc_stream", "load",
                                  span_props{{"ref_count", std::to_string(ref_count)}, {"mode_name", model_name}});

    dmlc::Stream *fi(dmlc::Stream::Create(params_path.c_str(), "r", true));
    if (fi == nullptr) {
      const auto msg = fmt::format("unable to create a stream for {}", params_path);
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
    stop_span(stream_span);

    NDArray::Load(fi, &arrays, &layer_names);

    // LOG(INFO) << "starting to convert " << arrays.size() << " ndarrays to
    // protobuf representation";

    const auto sharing_granularity = request->sharing_granularity();

    if (UPRD_PERSIST_CPU) {
      auto cpu_persist_span = start_span("persist_cpu", "load",
                                         span_props{{"ref_count", std::to_string(ref_count)},
                                                    {"mode_name", model_name},
                                                    {"granularity", SharingGranularity_Name(sharing_granularity)}});
      persist_on_cpu(sharing_granularity, model_name, arrays, layer_names);
      stop_span(cpu_persist_span);
    }

    auto layers_span = start_span("to_layers_from_disk", "load",
                                  span_props{{"ref_count", std::to_string(ref_count)},
                                             {"mode_name", model_name},
                                             {"granularity", SharingGranularity_Name(sharing_granularity)}});

    if (sharing_granularity == SharingGranularity_Layer) {
      size_t ii = 0;
      for (const auto &array : arrays) {
        const auto layer_name = layer_names[ii++];
        auto layer            = layers->Add();
        to_layer_from_disk(layer, layer_name, array, ref_count, stream);
      }
    } else if (sharing_granularity == SharingGranularity_Model) {

      if (is_persistent_on_cpu(model_name)) {
        auto e    = cpu_persistent_data.find(model_name);
        auto info = e->second;
        to_layers_from_model_info_for_model_granularity(layer_names, info, ref_count, stream);
      } else {
        model_granularity_to_layer_from_disk(layer_names, layers, arrays, ref_count, stream);
      }
    } else {
      throw std::runtime_error("invalid sharing granularity");
    }
    stop_span(layers_span);
  }

  void from_owned_layer(Layer *layer, const Layer &owned, int64_t ref_count) {

    const auto id    = sole::uuid4().str();
    const auto shape = layer->mutable_shape();

    auto span = start_span("from_owned_layer", "load",
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
    layer->set_offset(owned.offset());
    layer->set_device_raw_ptr(owned.device_raw_ptr());
    layer->set_ref_count(ref_count);
  }

  void from_owned_modelhandle(ModelHandle *handle, const ModelHandle &owned, int64_t ref_count) {

    const auto uuid = sole::uuid4().str();

    auto span = start_span("from_owned_modelhandle", "load",
                           span_props{{"ref_count", std::to_string(ref_count)}, {"model_id", owned.model_id()}});
    defer(stop_span(span));

    handle->set_id(uuid);
    handle->set_model_id(owned.model_id());
    handle->set_byte_count(owned.byte_count());
    handle->set_sharing_granularity(owned.sharing_granularity());
    handle->set_device_raw_ptr(owned.device_raw_ptr());
    handle->set_ipc_handle(owned.ipc_handle());
    handle->set_name(owned.name());
    handle->set_needed_eviction(owned.needed_eviction());

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
    const auto internal_memory_usage  = get_model_internal_memory_usage(model_name);

    auto span = start_span("estimate_model_size", "load",
                           span_props{{"estimation_rate", std::to_string(estimation_rate)},
                                      {"model_name", request->name()},
                                      {"internal_memory_usage", std::to_string(internal_memory_usage)}});
    defer(stop_span(span));

    std::ifstream in(params_path, std::ifstream::ate | std::ifstream::binary);
    return in.tellg() * estimation_rate + internal_memory_usage;
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
      auto model            = min_element->second;
      const auto byte_count = model->owned_model().byte_count();
      memory_usage_ -= byte_count;
      memory_freed += byte_count;
      model_delete(model);
      memory_db_.erase(min_element);
    }

    if (memory_freed >= memory_to_free) {
      ret = true;
    }

    return ret;
  }

  bool perform_fifo_eviction(const ModelRequest *request, const size_t memory_size_request,
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
      auto model            = min_element->second;
      const auto byte_count = model->owned_model().byte_count();
      memory_usage_ -= byte_count;
      memory_freed += byte_count;
      model_delete(model);
      memory_db_.erase(min_element);
    }

    if (memory_freed >= memory_to_free) {
      ret = true;
    }

    return ret;
  }

  bool perform_flush_eviction(const ModelRequest *request, const size_t memory_size_request,
                              const size_t memory_to_free) {
    size_t memory_freed = 0;
    for (const auto &elem : memory_db_) {
      auto model            = elem.second;
      const auto byte_count = model->owned_model().byte_count();
      memory_usage_ -= byte_count;
      memory_freed += byte_count;
      model_delete(model);
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
      auto model            = min_element->second;
      const auto byte_count = model->owned_model().byte_count();
      memory_usage_ -= byte_count;
      memory_freed += byte_count;
      model_delete(model);
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

    auto span = start_span("perform_eviction", "evict",
                           span_props{{"policy", eviction_policy},
                                      {"model_name", request->name()},
                                      {"estimated_model_size", std::to_string(estimated_model_size)},
                                      {"memory_to_free", std::to_string(memory_to_free)}});
    defer(stop_span(span));

    LOG(INFO) << "performing " << eviction_policy << " to get " << memory_to_free
              << " of extra memory for the estimated model size " << estimated_model_size;

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

  bool evict_if_needed(const ModelRequest *request) {
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
      return true;
    }
    return false;
  }

  size_t fifo_order{0};

public:
  // control memory usage by percentage of gpu
  grpc::Status Open(grpc::ServerContext *context, const ModelRequest *request, ModelHandle *reply) override {
    auto span = start_span("open", "grpc", span_props{{"model_name", request->name()}});
    defer(stop_span(span));

    // LOG(INFO) << "opening " << request->name();

    auto it = memory_db_.find(request->name());
    if (it == memory_db_.end()) {
      const auto uuid = sole::uuid4().str();

      cudaStream_t stream;
      CUDA_CHECK_CALL(cudaStreamCreate(&stream), "unable to create stream");

      auto owned_span = start_span("load_owned", "load", span_props{{"model_name", request->name()}});
      defer(stop_span(owned_span));

      bool needed_eviction = false;
      try {
        needed_eviction = evict_if_needed(request);
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
      owned_model->set_name(name);
      owned_model->set_needed_eviction(needed_eviction);

      load_ndarray(owned_model->mutable_layer(), request, /*ref_count=*/-1, stream);

      int64_t byte_count = 0;
      for (const auto it : owned_model->layer()) {
        byte_count += it.byte_count();
      }

      owned_model->set_byte_count(byte_count);
      owned_model->set_sharing_granularity(request->sharing_granularity());

      if (request->sharing_granularity() == SharingGranularity_Model) {
        const auto first_layer = owned_model->layer().first();
        owned_model->set_device_raw_ptr(first_layer.device_raw_ptr());
        owned_model->set_ipc_handle(first_layer.ipc_handle());
      } else {
        owned_model->set_device_raw_ptr(0);
        owned_model->set_ipc_handle("");
      }

      model->set_fifo_order(fifo_order++);

      memory_db_.insert({request->name(), model});
      memory_usage_ += byte_count;

      CUDA_CHECK_CALL(cudaStreamSynchronize(stream), "failed to synchronize stream");
      CUDA_CHECK_CALL(cudaStreamDestroy(stream), "failed to destroy stream");
    }

    auto shared_span = start_span("make_shared", "share", span_props{{"model_name", request->name()}});
    defer(stop_span(shared_span));

    // LOG(INFO) << "done with creating owned model";

    it = memory_db_.find(request->name());
    CHECK(it != memory_db_.end()) << "expecting the model to be there";

    auto model = it->second;
    CHECK(model != nullptr) << "expecting a valid model";

    // now we need to use the owned array to create
    // new memory handles
    // LOG(INFO) << "creating shared handle from owned memory";
    model->set_ref_count(model->ref_count() + 1);
    auto handle = model->mutable_shared_model()->Add();
    from_owned_modelhandle(handle, model->owned_model(), model->ref_count());
    // LOG(INFO) << "sending " << model->owned_model().layer().size() << "
    // layers to client";

    // LOG(INFO) << "finished satisfying open request";

    auto h = model->mutable_use_history()->Add();
    h->CopyFrom(TimeUtil::GetCurrentTime());

    auto t = model->mutable_lru_timestamp();
    t->CopyFrom(TimeUtil::GetCurrentTime());

    reply->CopyFrom(*handle);

    return grpc::Status::OK;
  }

  grpc::Status Info(grpc::ServerContext *context, const ModelRequest *request, Model *reply) override {

    auto span = start_span("info", "grpc", span_props{{"model_name", request->name()}});
    defer(stop_span(span));

    auto it = memory_db_.find(request->name());
    if (it == memory_db_.end()) {
      LOG(ERROR) << "failed to info request. cannot find " << request->name() << " in cache. "
                 << " cache = " << keys(memory_db_) << " \n";
      return grpc::Status(grpc::NOT_FOUND,
                          std::string("unable to find handle with name ") + request->name() + " during info request");
    }

    reply->CopyFrom(*it->second);

    return grpc::Status::OK;
  }

  std::string find_model_name_by_model_id(std::string model_id) {
    for (const auto &kv : memory_db_) {
      const auto &k = kv.first;
      const auto &v = kv.second;
      if (v->id() == model_id) {
        return k;
      }
    }
    return "";
  }

  void destroy_model_handle(const ModelHandle &handle) {
  }

  grpc::Status Close(grpc::ServerContext *context, const ModelHandle *request, Void *reply) override {

    auto span = start_span("close", "grpc", span_props{{"id", request->id()}, {"model_id", request->model_id()}});
    defer(stop_span(span));

    const auto model_name = find_model_name_by_model_id(request->model_id());
    if (model_name == "") {
      LOG(ERROR) << "failed to close request.  unable to find model name with id " << request->model_id()
                 << " during close request";
      return grpc::Status(grpc::NOT_FOUND,
                          std::string("unable to find model name with id ") + request->model_id() +
                              " during close request");
    }
    auto model_entry = memory_db_.find(model_name);
    if (model_entry == memory_db_.end()) {
      LOG(ERROR) << "failed to close request\n";
      return grpc::Status(grpc::NOT_FOUND,
                          std::string("unable to find handle with name ") + model_name + " during close request");
    }

    auto model              = model_entry->second;
    const auto handle_id    = request->id();
    const auto shared_model = model->mutable_shared_model();
    for (auto it = shared_model->begin(); it != shared_model->end(); it++) {
      auto shared_model_handle = *it;
      if (shared_model_handle.id() != handle_id) {
        continue;
      }
      destroy_model_handle(shared_model_handle);
      model->mutable_shared_model()->erase(it);
      break;
    }

    const auto ref_count = model->ref_count() - 1;
    model->set_ref_count(ref_count);

    if (ref_count == 0) {
      static const auto eviction_policy = UPRD_EVICTION_POLICY;
      if (eviction_policy == "eager" || UPRD_PERSIST_ONLY_CPU) {
        const auto byte_count = model->owned_model().byte_count();
        memory_usage_ -= byte_count;
        model_delete(model);
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
            << "max_memory_to_use = " << max_memory_to_use;
  if (UPRD_WRITE_PROFILE) {
    LOG(INFO) << "profile_path = " << profile_path;
  }

  force_runtime_initialization();

  MXPredInit();

  MXSetProfilerConfig(1, profile_path.c_str());

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
