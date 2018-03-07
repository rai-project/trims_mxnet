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
using namespace std::string_literals;
using namespace grpc;

namespace upr {

std::string server::host_name = "localhost"s;
int server::port              = dmlc::GetEnv("PORT", 50051);
std::string server::address   = fmt::format("{}:{}", host_name, port);

static TShape to_shape(Shape shape) {
  auto dim = shape.dim();
  TShape res(dim.begin(), dim.end());
  return res;
}

static void *get_device_ptr(const Layer &layer) {
  const auto ipc_handle = layer.ipc_handle();
  if (ipc_handle == "") {
    const auto msg = fmt::format("unable to get device ptr from {}. make sure handle is not empty", ipc_handle);
    LOG(FATAL) << msg;
    throw dmlc::Error(msg);
  }

  cudaIpcMemHandle_t handle;
  memcpy((uint8_t *) &handle, ipc_handle.c_str(), sizeof(handle));

  LOG(INFO) << "get handle = " << handle << "get base64 handle = " << utils::base64_encode(ipc_handle);

  void *device_ptr;
  auto span = start_span("cudaIpcOpenMemHandle", span_category_ipc, span_props{{"layer", layer.name()}});
  CUDA_CHECK_CALL(cudaIpcOpenMemHandle((void **) &device_ptr, handle, cudaIpcMemLazyEnablePeerAccess),
                  fmt::format("failed to open cuda ipc mem handle from {}", utils::base64_encode(ipc_handle)));
  stop_span(span);

  // LOG(INFO) << "get device_ptr = " << device_ptr;

  return device_ptr;
}

static void to_ndarray(std::vector<NDArray> *arrays, const Layer &layer) {
  const auto ctx = get_ctx();

  auto span = start_span("to_nd_array"s, span_category_serialization, span_props{{"layer", layer.name()}});
  defer(stop_span(span));

  const auto shape    = to_shape(layer.shape());
  const auto dev_mask = ctx.dev_mask();
  const auto dev_id   = ctx.dev_id;

  LOG(INFO) << "in layer=" << layer.name() << " getting device ptr using ctx = " << ctx;

  auto device_ptr = get_device_ptr(layer);

  auto span_creating =
      start_span("creating_nd_array"s, span_category_serialization, span_props{{"layer", layer.name()}});
  defer(stop_span(span_creating));

  TBlob blob(device_ptr, shape, dev_mask, dev_id);
  arrays->emplace_back(blob, dev_id, /* is_shared = */ true);

  return;
}

static void to_ndarrays(std::vector<NDArray> *arrays, std::vector<std::string> *keys, const ModelHandle &reply) {
  const auto layers = reply.layer();

  // LOG(INFO) << "got " << layers.size() << " layers form reply, before to_ndarray";

  for (const auto layer : layers) {
    keys->emplace_back(layer.name());
    to_ndarray(arrays, layer);
  }

  // LOG(INFO) << "finished nd_array conversion";

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
    auto span_loading = start_span("open", span_category_load, span_props{{"model_name", model_name}});
    defer(stop_span(span_loading));
    auto client           = client::get_connection();
    const auto open_reply = client->Open(model_name); // The actual RPC call!

    // LOG(INFO) << "Client received open reply: " << open_reply.id();

    auto span_converting = start_span("convering_to_nd_array", span_category_serialization);
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

} // namespace upr
#endif // MXNET_USE_CUDA
