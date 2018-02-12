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

    Model Info(const std::string &directory_path) {
      ModelRequest request;
      request.set_directory_path(directory_path);
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

    ModelHandle Open(const std::string &directory_path) {
      ModelRequest request;
      request.set_directory_path(directory_path);
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

  static void Load(std::string model_name, dmlc::Stream *fi,
                   std::vector<NDArray> *data, std::vector<std::string> *keys) {

    RegistryClient client(grpc::CreateChannel(
        server_address, grpc::InsecureChannelCredentials()));
    auto open_reply = client.Open(model_name); // The actual RPC call!

    std::cout << "Client received open reply: " << open_reply.id() << "\n";
  }
};


std::string client::server_host_name = server::host_name;
int client::server_port = server::port;
std::string client::server_address = server::address;

void Load(std::string model_name, dmlc::Stream *fi,
          std::vector<NDArray> *data, std::vector<std::string> *keys) {


  LOG(INFO) << "UPR:: loading in Client mode";

  client::Load(model_name, fi, data, keys);
  return;
}
} // namespace upr

