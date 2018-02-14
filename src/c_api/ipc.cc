#include "ipc.h"

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

TShape to_shape(Shape shape) {
  auto dim = shape.dim();
  TShape res(dim.begin(), dim.end());
  return res;
}

cudaIpcMemHandle_t get_cuda_ipc_mem_handle(const std::string &ipc_handle_path) {
  cudaIpcMemHandle_t handle;
  uint8_t buffer[sizeof(handle) + 1];
  memset(buffer, 0, sizeof(handle) + 1);

  FILE *fp = fopen(ipc_handle_path.c_str(), "r");
  if (fp == nullptr) {
    const auto msg = fmt::format("unable to open file at {}.", ipc_handle_path);
    LOG(FATAL) << msg;
    throw std::runtime_error(msg);
  }

  for (size_t ii = 0; ii < sizeof(handle); ii++) {
    const auto ret = fscanf(fp, "%c", buffer + ii);
    if (ret == EOF) {
      LOG(INFO) << "got EOF while reading " << ipc_handle_path;
    }
    if (ret != 1) {
      const auto msg =
          fmt::format("failed to read {} at {}.", ipc_handle_path, ii);
      LOG(FATAL) << msg;
      throw std::runtime_error(msg);
    }
  }

  memcpy((uint8_t *)&handle, buffer, sizeof(handle));

  fclose(fp);

  return handle;
}

void *get_device_ptr(const Layer &layer) {
  const auto ipc_handle_path = layer.ipc_handle_path();
  if (!file_exists(ipc_handle_path)) {
    const auto msg = fmt::format(
        "unable to get device ptr from {}. make sure the path exists",
        ipc_handle_path);
    LOG(FATAL) << msg;
    throw std::runtime_error(msg);
  }

  cudaIpcMemHandle_t handle = get_cuda_ipc_mem_handle(ipc_handle_path);

  void *device_ptr = nullptr;
  CUDA_CHECK_CALL(
      cudaIpcOpenMemHandle((void **)&device_ptr, handle,
                           cudaIpcMemLazyEnablePeerAccess),
      fmt::format("failed to open cuda ipc mem handle at {}", ipc_handle_path));

  return device_ptr;
}

NDArray to_ndarray(const Layer &layer) {
  const auto ctx = Context::GPU();

  const auto shape = to_shape(layer.shape());
  const auto dev_mask = ctx.dev_mask();
  const auto dev_id = ctx.dev_id;

  auto device_ptr = get_device_ptr(layer);

  TBlob blob(device_ptr, shape, dev_mask, dev_id);
  NDArray array(blob, dev_id);

  return array;
}
std::tuple<std::vector<NDArray>, std::vector<std::string>>
to_ndarrays(const ModelHandle &reply) {
  std::vector<NDArray> arrays{};
  std::vector<std::string> keys{};

  const auto layers = reply.layer();

  for (const auto layer : layers) {
    keys.emplace_back(layer.name());
    arrays.emplace_back(to_ndarray(layer));
  }

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
      const auto status = stub_->Info(&context, request, &reply);

      if (!status.ok()) {
        throw std::runtime_error(
            fmt::format("Error: [{}] {}. Info failed on client.",
                        status.error_message(), status.error_details()));
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
      const auto status = stub_->Open(&context, request, &reply);

      if (!status.ok()) {
        throw std::runtime_error(
            fmt::format("Error: [{}] {}. Open failed on client.",
                        status.error_message(), status.error_details()));
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

      const auto status = stub_->Close(&context, request, &reply);

      if (!status.ok()) {
        throw std::runtime_error(
            fmt::format("Error: [{}] {}. Close failed on client.",
                        status.error_message(), status.error_details()));
      }
      return;
    }

  private:
    std::unique_ptr<Registry::Stub> stub_;
  };

  static void Load(std::string model_name, std::vector<NDArray> *res_arrays,
                   std::vector<std::string> *res_keys) {

    RegistryClient client(grpc::CreateChannel(
        server_address, grpc::InsecureChannelCredentials()));

    const auto open_reply = client.Open(model_name); // The actual RPC call!

    LOG(INFO) << "Client received open reply: " << open_reply.id();

    const auto arrays_keys = to_ndarrays(open_reply);

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
