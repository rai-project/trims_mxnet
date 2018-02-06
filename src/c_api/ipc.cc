#include "ipc.h"

#include "fmt/format.h"

namespace upr {

struct server {
  // NDArray::Load();
  static const std::string host_name = "localhost";
  static const int port = dmlc::GetEnv("PORT", 50051);
  static const std::string address = fmt::format("{}:{}", host_name, port);
};

struct client {
  static const auto server_host_name = server::host_name;
  static const auto server_port = server::port;
  static const auto server_address = server::address;
};

void Load(dmlc::Stream *fi, std::vector<NDArray> *data,
          std::vector<std::string> *keys) {
  const bool is_server = dmlc::GetEnv("UPR_SERVER", true);
  LOG(INFO) << "UPR:: loading in " << (is_server ? "Server" : "Client")
            << " mode";

  if (is_server) {
    return server::Load(fi, data, keys);
  }
  return client::Load(fi, data, keys);
}
} // namespace upr