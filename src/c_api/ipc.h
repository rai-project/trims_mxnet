#pragma once
#ifdef MXNET_USE_CUDA

#include "driver_types.h"
#include <iostream>
#include <map>
#include <memory>
#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/engine.h>
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

#include "./../engine/profiler.h"
#include "./base64.h"
#include "./defer.h"
#include "fmt/format.h"
#include "prettyprint.hpp"

// - apt-get install binutils-dev ...
// - g++/clang++ -lbfd ...
#define BACKWARD_HAS_BFD 1

#include "backward.hpp"

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
      throw dmlc::Error(                                                       \
          fmt::format("CUDA[{}]:: {}", msg, cudaGetErrorString(e)));           \
    }                                                                          \
  }

static std::ostream &operator<<(std::ostream &os,
                                const cudaIpcMemHandle_t &handle) {
  const auto reserved = handle.reserved;
  for (int ii = 0; ii < CUDA_IPC_HANDLE_SIZE; ii++) {
    os << (int)reserved[ii];
  }
  return os;
}

namespace upr {
using namespace mxnet;

static const auto HOME = dmlc::GetEnv("HOME", std::string("/home/abduld"));
static const auto is_client = dmlc::GetEnv("UPR_CLIENT", false);
static const auto IPC_HANDLES_BASE_PATH = std::string("/tmp/persistent");
static const auto CARML_HOME_BASE_DIR =
    HOME + std::string("/carml/data/mxnet/");

static std::map<std::string, std::string> model_directory_paths{
    {"alexnet", CARML_HOME_BASE_DIR + "alexnet"},
    {"squeezenet", CARML_HOME_BASE_DIR + "squeezenetv1"},
    {"squeezenetv1", CARML_HOME_BASE_DIR + "squeezenetv1"},
    {"squeezenetv1.1", CARML_HOME_BASE_DIR + "squeezenetv1.1"},
    {"resnet-152-11k", CARML_HOME_BASE_DIR + "resnet-152-11k"},
    {"vgg16", CARML_HOME_BASE_DIR + "vgg16"}};


/**
 * @brief Ensures the CUDA runtime has fully initialized
 *
 * @note The CUDA runtime uses lazy initialization, so that until you perform
 * certain actions, the CUDA driver is not used to create a context, nothing
 * is done on the device etc. This function forces this initialization to
 * happen immediately, while not having any other effect.
 */
static inline void force_runtime_initialization() {
  // nVIDIA's Robin Thoni (https://www.rthoni.com/) guarantees
  // the following code "does the trick"
  CUDA_CHECK_CALL(cudaFree(nullptr), "Forcing CUDA runtime initialization");
}

static Context get_ctx() {
  static const auto ctx = Context::GPU();
  return ctx;
}

static int span_category_init = 5;
static int span_category_load = 6;
static int span_category_serialization = 7;
static int span_category_ipc = 8;
static int span_category_grpc = 9;

static engine::OprExecStat *start_span(const std::string &name, int category = 0) {
#if MXNET_USE_PROFILER
  const auto ctx = get_ctx();
  auto opr_stat = engine::Profiler::Get()->AddOprStat(ctx.dev_type, ctx.dev_id);
  uint64_t tid = std::hash<std::thread::id>()(std::this_thread::get_id());
  opr_stat->thread_id = tid;
  strncpy(opr_stat->opr_name, name.c_str(), name.size());
  opr_stat->opr_name[name.size()] = '\0';
  engine::SetOprCategory(opr_stat, category);
  engine::SetOprStart(opr_stat);
  return opr_stat;
#else
  return nullptr;
#endif
}

static void stop_span(engine::OprExecStat *stat) {
  if (stat == nullptr) {
    return;
  }

#if MXNET_USE_PROFILER
  engine::SetOprEnd(stat);
#endif
}

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

static std::string get_synset_path(std::string model_name = "") {
  const std::string path = get_model_directory_path(model_name);
  return path + "/synset.txt";
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
} // namespace upr
#endif // MXNET_USE_CUDA
