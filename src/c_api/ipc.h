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

#define DEFAULT_MODEL "bvlc_alexnet_1.0"

#ifdef NDEBUG
#define CUDA_CHECK_CALL(func, msg) func
#else
#define CUDA_CHECK_CALL(func, msg)                                                                                     \
  {                                                                                                                    \
    cudaError_t e = (func);                                                                                            \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) << "CUDA[" << msg << "]:: " << cudaGetErrorString(e);     \
    if (e != cudaSuccess) {                                                                                            \
      throw dmlc::Error(fmt::format("CUDA[{}]:: {}", msg, cudaGetErrorString(e)));                                     \
    }                                                                                                                  \
  }
#endif

static std::ostream &operator<<(std::ostream &os, const cudaIpcMemHandle_t &handle) {
  const auto reserved = handle.reserved;
  for (int ii = 0; ii < CUDA_IPC_HANDLE_SIZE; ii++) {
    os << (int) reserved[ii];
  }
  return os;
}

namespace upr {
using namespace mxnet;

static const auto HOME         = dmlc::GetEnv("HOME", std::string("/home/abduld"));
static const auto is_client    = dmlc::GetEnv("UPR_CLIENT", false);
static const auto UPR_BASE_DIR = dmlc::GetEnv("UPR_BASE_DIR", HOME + std::string("/carml/data/mxnet/"));

static std::map<std::string, std::string> model_directory_paths{{"bvlc_alexnet_1.0", UPR_BASE_DIR + "bvlc_alexnet_1.0"},
                                                                {"bvlc_googlenet_1.0", UPR_BASE_DIR + "bvlc_googlenet_1.0"},
                                                                {"bvlc_reference_caffenet_1.0", UPR_BASE_DIR + "bvlc_reference_caffenet_1.0"},
                                                                {"bvlc_reference_rcnn_ilsvrc13_1.0", UPR_BASE_DIR + "bvlc_reference_rcnn_ilsvrc13_1.0"},
                                                                {"dpn68_1.0", UPR_BASE_DIR + "dpn68_1.0"},
                                                                {"dpn92_1.0", UPR_BASE_DIR + "dpn92_1.0"},
                                                                {"inception_bn_3.0", UPR_BASE_DIR + "inception_bn_3.0"},
                                                                {"inception_resnet_2.0", UPR_BASE_DIR + "inception_resnet_2.0"},
                                                                {"inception_3.0", UPR_BASE_DIR + "inception_3.0"},
                                                                {"inception_4.0", UPR_BASE_DIR + "inception_4.0"},
                                                                {"inceptionbn_21k_1.0", UPR_BASE_DIR + "inceptionbn_21k_1.0"},
                                                                {"locationnet_1.0", UPR_BASE_DIR + "locationnet_1.0"},
                                                                {"network_in_network_1.0", UPR_BASE_DIR + "network_in_network_1.0"},
                                                                {"o_resnet101_2.0", UPR_BASE_DIR + "o_resnet101_2.0"},
                                                                {"o_resnet152_2.0", UPR_BASE_DIR + "o_resnet152_2.0"},
                                                                {"o_vgg16_1.0", UPR_BASE_DIR + "o_vgg16_1.0"},
                                                                {"o_vgg19_1.0", UPR_BASE_DIR + "o_vgg19_1.0"},
                                                                {"resnet18_2.0", UPR_BASE_DIR + "resnet18_2.0"},
                                                                {"resnet34_2.0", UPR_BASE_DIR + "resnet34_2.0"},
                                                                {"resnet50_2.0", UPR_BASE_DIR + "resnet50_2.0"},
                                                                {"resnet50_1.0", UPR_BASE_DIR + "resnet50_1.0"},
                                                                {"resnet101_2.0", UPR_BASE_DIR + "resnet101_2.0"},
                                                                {"resnet101_1.0", UPR_BASE_DIR + "resnet101_1.0"},
                                                                {"resnet152_11k_1.0", UPR_BASE_DIR + "resnet152_11k_1.0"},
                                                                {"resnet152_1.0", UPR_BASE_DIR + "resnet152_1.0"},
                                                                {"resnet152_2.0", UPR_BASE_DIR + "resnet152_2.0"},
                                                                {"resnet200_2.0", UPR_BASE_DIR + "resnet200_2.0"},
                                                                {"resnet269_2.0", UPR_BASE_DIR + "resnet269_2.0"},
                                                                {"resnext26_32x4d_1.0", UPR_BASE_DIR + "resnext26_32x4d_1.0"},
                                                                {"resnext50_32x4d_1.0", UPR_BASE_DIR + "resnext50_32x4d_1.0"},
                                                                {"resnext50_1.0", UPR_BASE_DIR + "resnext50_1.0"},                                                                
                                                                {"resnext101_32x4d_1.0", UPR_BASE_DIR + "resnext101_32x4d_1.0"},
                                                                {"resnext101_1.0", UPR_BASE_DIR + "resnext101_1.0"},
                                                                {"squeezenet_1.0", UPR_BASE_DIR + "squeezenet_1.0"},
                                                                {"squeezenet_1.1", UPR_BASE_DIR + "squeezenet_1.1"},
                                                                {"vgg16_sod_1.0", UPR_BASE_DIR + "vgg16_sod_1.0"},                                                                
                                                                {"vgg16_sos_1.0", UPR_BASE_DIR + "vgg16_sos_1.0"},                                                                
                                                                {"vgg16_1.0", UPR_BASE_DIR + "vgg16_1.0"},                                                                
                                                                {"vgg19_1.0", UPR_BASE_DIR + "vgg19_1.0"},                                                                
                                                                {"xception_1.0", UPR_BASE_DIR + "xception_1.0"},                                                                
                                                                {"wrn50_2.0", UPR_BASE_DIR + "wrn50_2.0"}};

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

static std::string span_category_init          = "init";
static std::string span_category_load          = "load";
static std::string span_category_serialization = "serialization";
static std::string span_category_ipc           = "ipc";
static std::string span_category_grpc          = "grpc";
static std::string span_category_mxnet_init    = "init";
static std::string span_category_ignore    = "ignore";

using span_props = std::map<std::string, std::string>;

static inline engine::OprExecStat *start_span(const std::string &name, std::string category ) {
#if MXNET_USE_PROFILER
  const auto ctx = get_ctx();
  auto opr_stat  = engine::Profiler::Get()->AddOprStat(ctx.dev_type, ctx.dev_id);
  uint64_t tid   = std::hash<std::thread::id>()(std::this_thread::get_id());
  opr_stat->opr_name = name;
  engine::SetOprCategory(opr_stat, category);
  engine::SetOprStart(opr_stat);
  return opr_stat;
#else
  return nullptr;
#endif
}

static inline engine::OprExecStat *start_span(const std::string &name, 
                                              std::string category, span_props props) {
#if MXNET_USE_PROFILER
  auto span = start_span(name, category);
  for (const auto kv : props) {
    engine::AddOprMetadata(span, kv.first, kv.second);
  }
  return span;
#else
  return nullptr;
#endif
}

static inline void stop_span(engine::OprExecStat *stat) {
  if (stat == nullptr) {
    return;
  }

#if MXNET_USE_PROFILER
  engine::SetOprEnd(stat);
#endif
}

static std::string get_model_name() {
  static const auto model_name = dmlc::GetEnv("UPR_MODEL_NAME", std::string(DEFAULT_MODEL));
  return model_name;
}

static std::string get_model_directory_path(std::string model_name = "") {
  if (model_name == "") {
    model_name = get_model_name();
  }
  const auto it = model_directory_paths.find(model_name);
  if (it == model_directory_paths.end()) {
    throw dmlc::Error(fmt::format("unable to find {} model in model_direction_paths", model_name));
  }
  return it->second;
}

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

static std::string get_model_params_path(std::string model_name = "") {
  if (model_name == "") {
    model_name = get_model_name();
  }
  const std::string path = get_model_directory_path(model_name);
  auto model_path        = path + "/model.params";
  if (file_exists(model_path)) {
    return model_path;
  }
  model_path = path + "/" + model_name + ".params";
  if (file_exists(model_path)) {
    return model_path;
  }
  model_path = path + "/" + model_name + "-0000.params";
  if (file_exists(model_path)) {
    return model_path;
  }

  throw dmlc::Error(fmt::format("unable to find {} model params in model_directory_path. make sure "
                                "that you have the model"
                                "in the directory and it's called either model.params, "
                                "{}.params, or {}-0000.params",
                                model_name, model_name, model_name));

  return "";
}

static std::string get_model_symbol_path(std::string model_name = "") {
  if (model_name == "") {
    model_name = get_model_name();
  }
  const std::string path = get_model_directory_path(model_name);
  auto model_path        = path + "/model.symbol";
  if (file_exists(model_path)) {
    return model_path;
  }
  model_path = path + "/" + model_name + ".symbol";
  if (file_exists(model_path)) {
    return model_path;
  }
  model_path = path + "/" + model_name + "-symbol.json";
  if (file_exists(model_path)) {
    return model_path;
  }

  throw dmlc::Error(fmt::format("unable to find {} model symbol in model_directory_path. make sure "
                                "that you have the model"
                                "in the directory and it's called either model.symbol, "
                                "{}.symbol, or {}-symbol.json",
                                model_name, model_name, model_name));

  return "";
}

static std::string get_synset_path(std::string model_name = "") {
  if (model_name == "") {
    model_name = get_model_name();
  }
  const std::string path = get_model_directory_path(model_name);

  auto synset_path = path + "/features.txt";
  if (file_exists(synset_path)) {
    return synset_path;
  }

  throw dmlc::Error(fmt::format("unable to find {} model synset in {}. make sure "
                                "that you have the synset file in the directory and it's called synset.txt",
                                model_name, path));

  return "";
}

struct server {
  static std::string host_name;
  static int port;
  static std::string address;
};

template <typename charT>
inline bool string_starts_with(const std::basic_string<charT> &big, const std::basic_string<charT> &small) {
  const typename std::basic_string<charT>::size_type big_size   = big.size();
  const typename std::basic_string<charT>::size_type small_size = small.size();
  const bool valid_                                             = (big_size >= small_size);
  const bool starts_with_                                       = (big.compare(0, small_size, small) == 0);
  return valid_ and starts_with_;
}

void Load(std::string model_name, std::vector<mxnet::NDArray> *data, std::vector<std::string> *keys);
} // namespace upr
#endif // MXNET_USE_CUDA
