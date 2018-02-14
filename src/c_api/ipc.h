#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/io.h>
#include <mxnet/kvstore.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/rtc.h>
#include <mxnet/storage.h>
#include <string>
#include <vector>

#include <fcntl.h>
#include <stdexcept>
#include <thread>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "fmt/format.h"

#define BYTE 1
#define KBYTE 1024 * BYTE
#define MBYTE 1024 * KBYTE
#define DATA_SIZE 4 * BYTE

#define DEFAULT_MODEL "alexnet"

#define CUDA_CHECK_CALL(func, msg)                                             \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                   \
        << "CUDA[" << msg << "]:: " << cudaGetErrorString(e);                  \
    if (e != cudaSuccess) {                                                    \
      throw std::runtime_error(                                                \
          fmt::format("CUDA[{}]:: {}", msg, cudaGetErrorString(e)));           \
    }                                                                          \
  }

namespace upr {

static const auto IPC_HANDLES_BASE_PATH = std::string("/tmp/persistent");
static const auto CARML_HOME_BASE_DIR =
    std::string("/home/abduld/carml/data/mxnet/");

static std::map<std::string, std::string> model_directory_paths{
    {"alexnet", CARML_HOME_BASE_DIR + "alexnet"},
    {"squeezenet", CARML_HOME_BASE_DIR + "squeezenetv1"},
    {"squeezenetv1", CARML_HOME_BASE_DIR + "squeezenetv1"},
    {"squeezenetv1.1", CARML_HOME_BASE_DIR + "squeezenetv1.1"},
    {"vgg16", CARML_HOME_BASE_DIR + "vgg16"}};

static std::string get_model_name() {
  static const auto model_name =
      dmlc::GetEnv("UPR_MODEL_NAME", std::string(DEFAULT_MODEL));
  return model_name;
}

static std::string get_model_directory_path(std::string model_name = "") {
  if (model_name == "") {
    model_name = get_model_name();
  }
  const auto it = model_directory_paths.find(model_name);
  if (it == model_directory_paths.end()) {
    throw dmlc::Error(fmt::format(
        "unable to find {} model in model_direction_paths", model_name));
  }
  return it->second;
}

static std::string get_model_params_path(std::string model_name = "") {
  const std::string path = get_model_directory_path(model_name);
  return path + "/model.params";
}

static std::string get_model_symbol_path(std::string model_name = "") {
  const std::string path = get_model_directory_path(model_name);
  return path + "/model.symbol";
}

static std::string get_synset_path() {
  return CARML_HOME_BASE_DIR + "synset.txt";
}

struct server {
  static std::string host_name;
  static int port;
  static std::string address;
};

static bool directory_exists(const std::string &path) {
  struct stat sb;
  if (stat(path.c_str(), &sb) == -1) {
    return false;
  }

  if ((sb.st_mode & S_IFMT) == S_IFDIR) {
    return true;
  }
  return false;
}

static bool file_exists(const std::string &path) {

  struct stat sb;
  if (stat(path.c_str(), &sb) == -1) {
    return false;
  }

  if ((sb.st_mode & S_IFMT) == S_IFDIR) {
    return false;
  }
  return true;
}

template <typename charT>
inline bool string_starts_with(const std::basic_string<charT> &big,
                               const std::basic_string<charT> &small) {
  const typename std::basic_string<charT>::size_type big_size = big.size();
  const typename std::basic_string<charT>::size_type small_size = small.size();
  const bool valid_ = (big_size >= small_size);
  const bool starts_with_ = (big.compare(0, small_size, small) == 0);
  return valid_ and starts_with_;
}

void Load(std::string model_name, std::vector<mxnet::NDArray> *data,
          std::vector<std::string> *keys);
}
