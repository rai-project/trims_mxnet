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

struct server {
  // NDArray::Load();
  static std::string host_name;
  static int port;
  static std::string address;
  static void initialize();

  static void Load(std::string symbol_json_str, dmlc::Stream *fi,
                   std::vector<NDArray> *data, std::vector<std::string> *keys) {
    return;
  }
};

struct client {
  static std::string server_host_name;
  static int server_port;
  static std::string server_address;
  static void initialize();
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

    Model Info(const std::string &file_path) {
      ModelRequest request;
      request.set_file_path(file_path);
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

    ModelHandle Open(const std::string &file_path) {
      ModelRequest request;
      request.set_file_path(file_path);
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

  static void Load(std::string symbol_json_str, dmlc::Stream *fi,
                   std::vector<NDArray> *data, std::vector<std::string> *keys) {

    RegistryClient Registry(grpc::CreateChannel(
        server_address, grpc::InsecureChannelCredentials()));
    auto open_reply = Registry.Open(symbol_json_str); // The actual RPC call!

    std::cout << "Client received open reply: " << open_reply.id() << "\n";
  }
};

void Load(std::string symbol_json_str, dmlc::Stream *fi,
          std::vector<NDArray> *data, std::vector<std::string> *keys) {

  server::initialize();
  client::initialize();

  const bool is_server = dmlc::GetEnv("UPR_SERVER", true);
  LOG(INFO) << "UPR:: loading in " << (is_server ? "Server" : "Client")
            << " mode";

  if (is_server) {
    server::Load(symbol_json_str, fi, data, keys);
    return;
  }
  client::Load(symbol_json_str, fi, data, keys);
  return;
}
} // namespace upr

void upr::server::initialize() {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }
  is_initialized = true;
  server::host_name = "localhost"s;
  server::port = dmlc::GetEnv("PORT", 50051);
  server::address = fmt::format("{}:{}", host_name, port);
}

void upr::client::initialize() {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }
  is_initialized = true;
  upr::server::initialize();
  client::server_host_name = server::host_name;
  client::server_port = server::port;
  client::server_address = server::address;
}