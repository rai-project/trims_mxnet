#ifdef MXNET_USE_CUDA
#include "ipc.h"

#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <dmlc/omp.h>
#include <dmlc/recordio.h>
#include <dmlc/type_traits.h>
#include <nnvm/node.h>

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

#include "./upr.grpc.pb.h"
#include "./upr.pb.h"

#include "fmt/format.h"

using namespace mxnet;
using namespace grpc;

namespace upr {

std::string server::host_name = "localhost";
int server::port              = dmlc::GetEnv("PORT", 50051);
std::string server::address   = fmt::format("{}:{}", host_name, port);


static TShape to_shape(Shape shape) {
  auto dim = shape.dim();
  TShape res(dim.begin(), dim.end());
  return res;
}

static void *get_device_ptr_offset(const Layer &layer, void *devPtr) {
  const auto offset = layer.offset();
  return (void *) (((char *) (devPtr)) + offset);
}

static void *get_device_ptr(const std::string &handle_bytes) {
  if (handle_bytes == "") {
    const auto msg = fmt::format("unable to get device ptr from {}. make sure handle is not empty", handle_bytes);
    LOG(FATAL) << msg;
    throw dmlc::Error(msg);
  }
  cudaIpcMemHandle_t handle;
  memcpy((uint8_t *) &handle, handle_bytes.c_str(), sizeof(handle));

  void *device_ptr = nullptr;
  CUDA_CHECK_CALL(cudaIpcOpenMemHandle((void **) &device_ptr, handle, cudaIpcMemLazyEnablePeerAccess),
                  fmt::format("failed to open cuda ipc mem handle from {}", utils::base64_encode(handle_bytes)));

  return device_ptr;
}

static void *get_device_ptr(const Layer &layer) {
  auto name             = layer.name();
  const auto ipc_handle = layer.ipc_handle();

  static const std::string arg_prefix("arg:");
  if (string_starts_with(name, arg_prefix)) {
    name.erase(0, arg_prefix.size());
  }
  static const std::string aux_prefix("aux:");
  if (string_starts_with(name, aux_prefix)) {
    name.erase(0, aux_prefix.size());
  }

  auto span       = start_span("cudaIpcOpenMemHandle",
                         span_category_ipc,
                         span_props{{"layer", name}, {"byte_count", std::to_string(layer.byte_count())}});
  auto device_ptr = get_device_ptr(ipc_handle.c_str());
  stop_span(span);

  return device_ptr;
}

static void to_ndarrays(std::vector<NDArray> *arrays, std::vector<std::string> *keys, const ModelHandle &model_handle) {
  const auto ctx      = get_ctx();
  const auto dev_mask = ctx.dev_mask();
  const auto dev_id   = ctx.dev_id;

  const auto layers = model_handle.layer();

  // LOG(INFO) << "got " << layers.size() << " layers form reply, before to_ndarray";

  if (model_handle.sharing_granularity() == SharingGranularity_Model) {
    auto ipc_open_span = start_span(
        "cudaIpcOpenMemHandle",
        span_category_ipc,
        span_props{{"model", model_handle.name()}, {"byte_count", std::to_string(model_handle.byte_count())}});
    auto base_device_ptr = get_device_ptr(model_handle.ipc_handle());
    stop_span(ipc_open_span);

    for (const auto layer : layers) {
     //auto create_layer_span = start_span("to_nd_array",
     //                                    span_category_serialization,
     //                                    span_props{{"layer", layer.name()}, {"sharing_granularity", "model"}});

      keys->emplace_back(layer.name());
      const auto shape = to_shape(layer.shape());
      auto device_ptr  = get_device_ptr_offset(layer, base_device_ptr);
      TBlob blob(device_ptr, shape, dev_mask, dev_id);
      arrays->emplace_back(blob, dev_id, /* is_shared = */ true);

      //stop_span(create_layer_span);
    }
    return;
  }
  if (model_handle.sharing_granularity() == SharingGranularity_Model) {
    for (const auto layer : layers) {
     //auto create_layer_span = start_span("to_nd_array",
     //                                    span_category_serialization,
     //                                    span_props{{"layer", layer.name()}, {"sharing_granularity", "layer"}});

      keys->emplace_back(layer.name());
      const auto shape = to_shape(layer.shape());
      auto device_ptr  = get_device_ptr(layer);
      TBlob blob(device_ptr, shape, dev_mask, dev_id);
      arrays->emplace_back(blob, dev_id, /* is_shared = */ true);

      //stop_span(create_layer_span);
    }
    return;
  }

  throw dmlc::Error("invalid granularity");

  return;
}

struct client {
  static std::string server_host_name;
  static int server_port;
  static std::string server_address;
  class RegistryClient {
  public:
    explicit RegistryClient(std::shared_ptr<Channel> channel) : stub_(Registry::NewStub(channel)) {
    }

    Model Info(const ModelRequest &request) {
      Model reply;
      ClientContext context;

      auto span = start_span("info", span_category_grpc);
      defer(stop_span(span));

      const auto status = stub_->Info(&context, request, &reply);

      if (!status.ok()) {
        throw dmlc::Error(
            fmt::format("Error: [{}] {}. Info failed on client.", status.error_message(), status.error_details()));
      }
      return reply;
    }

    Model Info(const std::string &model_name) {
      ModelRequest request;
      request.set_name(model_name);
      return this->Info(request);
    }

    ModelHandle Open(const ModelRequest &request) {
      ModelHandle reply;
      ClientContext context;

      auto span = start_span("open", span_category_grpc);
      defer(stop_span(span));

      const auto status = stub_->Open(&context, request, &reply);

      if (!status.ok()) {
        throw dmlc::Error(
            fmt::format("Error: [{}] {}. Open failed on client.", status.error_message(), status.error_details()));
      }
      return reply;
    }

    ModelHandle Open(const std::string &model_name) {
      ModelRequest request;
      request.set_name(model_name);
      if (UPR_SHARING_GRANULARITY == "model") {
        request.set_sharing_granularity(SharingGranularity_Model);
      } else if (UPR_SHARING_GRANULARITY == "layer") {
        request.set_sharing_granularity(SharingGranularity_Layer);
      } else {
        throw dmlc::Error(
            fmt::format("Error: [{}] {}. failed to determine model granularity.", UPR_SHARING_GRANULARITY));
      }
      return this->Open(request);
    }

    void Close(const ModelHandle &request) {
      Void reply;
      ClientContext context;

      auto span = start_span("close", span_category_grpc);
      defer(stop_span(span));

      const auto status = stub_->Close(&context, request, &reply);

      if (!status.ok()) {
        throw dmlc::Error(
            fmt::format("Error: [{}] {}. Close failed on client.", status.error_message(), status.error_details()));
      }
      return;
    }

    void Close(const std::string &handle_id, const std::string &model_id) {
      ModelHandle request;
      request.set_id(handle_id);
      request.set_model_id(model_id);
      return this->Close(request);
    }

  private:
    std::unique_ptr<Registry::Stub> stub_;
  };

  static RegistryClient *get_connection() {
    static RegistryClient *client =
        new RegistryClient(grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));
        std::cout << "initialize = true \n";
    return client;
  }

  static void Unload(MXAPIPredictor *pred) {
    auto span = start_span(
        "close", span_category_close, span_props{{"model_name", pred->model_name}, {"model_id", pred->model_id}});
    defer(stop_span(span));

    auto client = client::get_connection();
    client->Close(pred->handle_id, pred->model_id);

    return;
  }

  static std::pair<std::string, std::string>
      Load(std::string model_name, std::vector<NDArray> *res_arrays, std::vector<std::string> *res_keys) {
    auto span_loading = start_span("load_model", span_category_load, span_props{{"model_name", model_name}});
    defer(stop_span(span_loading));
    auto client           = client::get_connection();
    const auto open_reply = client->Open(model_name); // The actual RPC call!

    // LOG(INFO) << "Client received open reply: " << open_reply.id();

    auto span_converting = start_span("convering_to_nd_array",
                                      span_category_serialization,
                                      span_props{{"model_id", open_reply.model_id()},
                                                 {"byte_count", std::to_string(open_reply.byte_count())},
                                                 {"needed_eviction", std::to_string(open_reply.needed_eviction())},
                                                 {"nlayers", std::to_string(open_reply.layer().size())}});
    defer(stop_span(span_converting));

    to_ndarrays(res_arrays, res_keys, open_reply);

    // LOG(INFO) << "Loaded model " << model_name;

    return std::make_pair(open_reply.id(), open_reply.model_id());
  }
};

std::string client::server_host_name = server::host_name;
int client::server_port              = server::port;
std::string client::server_address   = server::address;

std::pair<std::string, std::string>
    Load(std::string model_name, std::vector<NDArray> *data, std::vector<std::string> *keys) {

  LOG(INFO) << "UPR:: loading in Client mode";

  return client::Load(model_name, data, keys);
}

void Unload(MXAPIPredictor *pred) {
  LOG(INFO) << "UPR:: closing in Client mode";
  client::Unload(pred);
  return;
}

void initialize() {
    if (is_client && UPR_ENABLED) {
        client::get_connection();
    }
}

} // namespace upr
#endif // MXNET_USE_CUDA
