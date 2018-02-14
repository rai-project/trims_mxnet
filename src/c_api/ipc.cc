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

  TShape to_shape(Shape shape) {
    auto dim = shape.dim();
    TShape res(dim.begin(), dim.end());
    return res;
  }

  NDArray to_ndarray(Layer layer) {
    const auto shape = to_shape(layer.shape());
    const auto ctx = Context::GPU();

    NDArray array(shape, ctx);

    return array;
  }
  void dump_ndarray(std::vector<NDArray> *data, std::vector<std::string> *keys,
                    const ModelHandle *reply) {
    auto layers = reply->layer();

    for (const auto layer : layers) {
    }
  }

  static void Load(std::string model_name, std::vector<NDArray> *data,
                   std::vector<std::string> *keys) {

    RegistryClient client(grpc::CreateChannel(
        server_address, grpc::InsecureChannelCredentials()));
    auto open_reply = client.Open(model_name); // The actual RPC call!
    if (!open_reply) {
      const auto msg = fmt::format("Error: {}", open_reply.error());
      LOG(ERROR) << msg;
    }

    dump_ndarray(data, keys, open_reply);
    std::cout << ")Client received open reply: " << open_reply.id() << "\n";
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
