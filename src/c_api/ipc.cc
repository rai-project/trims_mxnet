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
int server::port = dmlc::GetEnv("PORT", 50051);
std::string server::address = fmt::format("{}:{}", host_name, port);

static TShape to_shape(Shape shape) {
  auto dim = shape.dim();
  TShape res(dim.begin(), dim.end());
  return res;
}

static void *get_device_ptr(const Layer &layer) {
  const auto ipc_handle = layer.ipc_handle();
  if (ipc_handle == "") {
    const auto msg = fmt::format(
        "unable to get device ptr from {}. make sure handle is not empty",
        ipc_handle);
    LOG(FATAL) << msg;
    throw dmlc::Error(msg);
  }

  cudaIpcMemHandle_t handle;
  // memcpy((uint8_t *)&handle, ipc_handle.c_str(), sizeof(handle));
  // memcpy((uint8_t *)&handle, utils::base64_decode(ipc_handle).c_str(),
  // sizeof(handle));
  memcpy((uint8_t *)&handle, ipc_handle.c_str(), sizeof(handle));

  LOG(INFO) << "get handle = " << handle
            << "get base64 handle = " << utils::base64_encode(ipc_handle);


    auto span= start_span("performing cudaIpcOpenMemHandle for "s + layer.name() , span_category_ipc);
    defer(stop_span(span));

  // LOG(INFO) << "open cuda mem handle = " << handle;
  void *device_ptr;
#if 1
  CUDA_CHECK_CALL(cudaIpcOpenMemHandle((void **)&device_ptr, handle,
                                       cudaIpcMemLazyEnablePeerAccess),
                  fmt::format("failed to open cuda ipc mem handle from {}",
                              utils::base64_encode(ipc_handle)));
#else

  cudaIpcOpenMemHandle((void **)&device_ptr, handle,
                       cudaIpcMemLazyEnablePeerAccess),
#endif
  LOG(INFO) << "get device_ptr = " << device_ptr;

#if 0
  LOG(INFO) << "doing cudamemcpy using the layer " << layer.name();
  char buf[5];
  memset(buf, 0, 5);
  CUDA_CHECK_CALL(cudaMemcpy(buf, device_ptr, 5, cudaMemcpyDeviceToHost),
                  "cuda memcpy failed");
  for (int ii = 0; ii < 5; ii++) {
    LOG(INFO) << "for cuda memcpy at " << ii << " got the value "
              << (int)buf[ii];
  }
#endif
  return device_ptr;
}

static NDArray to_ndarray(const Layer &layer) {
  const auto ctx = get_ctx();

    auto span= start_span("convering "s + layer.name() +  " to  nd_array"s, span_category_serialization);
    defer(stop_span(span));

  const auto shape = to_shape(layer.shape());
  const auto dev_mask = ctx.dev_mask();
  const auto dev_id = ctx.dev_id;

  LOG(INFO) << "in layey=" << layer.name()
            << " getting device ptr using ctx = " << ctx;

  auto device_ptr = get_device_ptr(layer);

    auto span_creating = start_span("creating nd_array for "s + layer.name() , span_category_serialization);
    defer(stop_span(span_creating));

  TBlob blob(device_ptr, shape, dev_mask, dev_id);
  NDArray array(blob, dev_id);

  return array;
}
static std::tuple<std::vector<NDArray>, std::vector<std::string>>
to_ndarrays(const ModelHandle &reply) {
  std::vector<NDArray> arrays{};
  std::vector<std::string> keys{};

  const auto layers = reply.layer();

  LOG(INFO) << "got " << layers.size()
            << " layers form reply, before to_ndarray";

  for (const auto layer : layers) {
    keys.emplace_back(layer.name());
    arrays.emplace_back(to_ndarray(layer));
  }

  LOG(INFO) << "finished nd_array conversion";

  return std::make_tuple(arrays, keys);
}

struct client {
  static std::string server_host_name;
  static int server_port;
  static std::string server_address;
  class RegistryClient {
  public:
    explicit RegistryClient(std::shared_ptr<Channel> channel)
        : stub_(Registry::NewStub(channel)) {}

    Model Info(const ModelRequest &request) {
      Model reply;
      ClientContext context;

      auto span = start_span("grpc info", span_category_grpc);
      defer(stop_span(span));

      const auto status = stub_->Info(&context, request, &reply);

      if (!status.ok()) {
        throw dmlc::Error(fmt::format("Error: [{}] {}. Info failed on client.",
                                      status.error_message(),
                                      status.error_details()));
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

      auto span = start_span("grpc open", span_category_grpc);
      defer(stop_span(span));

      const auto status = stub_->Open(&context, request, &reply);

      if (!status.ok()) {
        throw dmlc::Error(fmt::format("Error: [{}] {}. Open failed on client.",
                                      status.error_message(),
                                      status.error_details()));
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

      auto span = start_span("grpc close", span_category_grpc);
      defer(stop_span(span));

      const auto status = stub_->Close(&context, request, &reply);

      if (!status.ok()) {
        throw dmlc::Error(fmt::format("Error: [{}] {}. Close failed on client.",
                                      status.error_message(),
                                      status.error_details()));
      }
      return;
    }

  private:
    std::unique_ptr<Registry::Stub> stub_;
  };

  static void Load(std::string model_name, std::vector<NDArray> *res_arrays,
                   std::vector<std::string> *res_keys) {
    auto span_loading = start_span("loading nd_array", span_category_load);
    defer(stop_span(span_loading));

    RegistryClient client(grpc::CreateChannel(
        server_address, grpc::InsecureChannelCredentials()));

    const auto open_reply = client.Open(model_name); // The actual RPC call!

    LOG(INFO) << "Client received open reply: " << open_reply.id();

    auto span_converting = start_span("convering to nd_array", span_category_serialization);
    defer(stop_span(span_converting));
    const auto arrays_keys = to_ndarrays(open_reply);

    LOG(INFO) << "Loaded model " << model_name;
    *res_arrays = std::get<0>(arrays_keys);
    *res_keys = std::get<1>(arrays_keys);
  }
};

std::string client::server_host_name = server::host_name;
int client::server_port = server::port;
std::string client::server_address = server::address;

void Load(std::string model_name, std::vector<NDArray> *data,
          std::vector<std::string> *keys) {

  LOG(INFO) << "UPR:: loading in Client mode";

  client::Load(model_name, data, keys);
  return;
}
} // namespace upr
