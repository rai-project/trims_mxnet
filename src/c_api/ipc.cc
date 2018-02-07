#include "ipc.h"

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

#include "./upr.grpc.pb.h"
#include "./upr.pb.h"

#include "fmt/format.h"

namespace upr {

struct server {
  // NDArray::Load();
  static const std::string host_name = "localhost";
  static const int port = dmlc::GetEnv("PORT", 50051);
  static const std::string address = fmt::format("{}:{}", host_name, port);
};

struct client {
  using grpc;
  static const auto server_host_name = server::host_name;
  static const auto server_port = server::port;
  static const auto server_address = server::address;
class RegistryClient {
public:
  explicit RegistryClient(std::shared_ptr<Channel> channel) : stub_(Registry::NewStub(channel)) {
  }

  Model Info(const ModelRequest & request) {

    // Container for the data we expect from the server.
    Model reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // Storage for the status of the RPC upon completion.
    Status status = stub_->Info(&context, request, &reply);

    // Act upon the status of the actual RPC.
    if (!status.ok()) {
      throw std::runtime_error(
        fmt::format(
          "Error: [{}] {}. Info failed on client.", 
        status.error_message(),
         status.error_details()
        )
        );
    }
    return reply;
  }

  Model Info(const std::string & file_path) {
    ModelRequest request;
    request.set_file_path(file_path);
    return this->Info(request);
  }
  
  ModelHandle Open(const ModelRequest & request) {

    // Container for the data we expect from the server.
    ModelHandle reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // Storage for the status of the RPC upon completion.
    Status status = stub_->Open(&context, request, &reply);

    // Act upon the status of the actual RPC.
    if (!status.ok()) {
      throw std::runtime_error(
        fmt::format(
          "Error: [{}] {}. Open failed on client.", 
        status.error_message(),
         status.error_details()
        )
        );
    }
    return reply;
  }

  ModelHandle Open(const std::string & file_path) {
    ModelRequest request;
    request.set_file_path(file_path);
    return this->Open(request);
  }

  void Close(const ModelHandle & handle)  {
    // Container for the data we expect from the server.
    Void reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // Storage for the status of the RPC upon completion.
    Status status = stub_->Close(&context, request, &reply);

    // Act upon the status of the actual RPC.
    if (!status.ok()) {
      throw std::runtime_error(
        fmt::format(
          "Error: [{}] {}. Close failed on client.", 
        status.error_message(),
         status.error_details()
        )
        );
    }
    return ;
  }

private:
  // Out of the passed in Channel comes the stub, stored here, our view of the
  // server's exposed services.
  std::unique_ptr<Registry::Stub> stub_;
};

static void Load(std::string symbol_json_str, dmlc::Stream *fi, std::vector<NDArray> *data,
          std::vector<std::string> *keys) {

  RegistryClient Registry(grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));
 auto open_reply = Registry.Open(symbol_json_str); // The actual RPC call!

  std::cout << "Client received open reply: " << open_reply.id() << "\n";
          }
};

void Load(std::string symbol_json_str, dmlc::Stream *fi, std::vector<NDArray> *data,
          std::vector<std::string> *keys) {
  const bool is_server = dmlc::GetEnv("UPR_SERVER", true);
  LOG(INFO) << "UPR:: loading in " << (is_server ? "Server" : "Client")
            << " mode";

  if (is_server) {
    return server::Load(symbol_json_str, fi, data, keys);
  }
  return client::Load(symbol_json_str, fi, data, keys);
}
} // namespace upr