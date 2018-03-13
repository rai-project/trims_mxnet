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
#include <tuple>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "./../engine/profiler.h"
#include "./base64.h"
#include "./c_api_common.h"
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

static const auto HOME           = dmlc::GetEnv("HOME", std::string("/home/abduld"));
static const auto UPR_ENABLED    = dmlc::GetEnv("UPR_ENABLED", true);
static const auto UPR_PROFILE_IO = !UPR_ENABLED && dmlc::GetEnv("UPR_PROFILE_IO", true);
static const auto is_client      = dmlc::GetEnv("UPR_CLIENT", false);
static const auto UPR_BASE_DIR   = dmlc::GetEnv("UPR_BASE_DIR", HOME + std::string("/carml/data/mxnet/"));

static const auto UPR_ENABLE_MEMORY_PROFILE = dmlc::GetEnv("UPR_ENABLE_MEMORY_PROFILE", false);
static const auto UPR_ENABLE_CUDA_FREE      = dmlc::GetEnv("UPR_ENABLE_CUDA_FREE", false);

static const auto UPRD_EVICTION_POLICY   = dmlc::GetEnv("UPRD_EVICTION_POLICY", std::string("lru"));
static const auto UPRD_ESTIMATION_RATE   = dmlc::GetEnv("UPRD_ESTIMATION_RATE", 3.0);
static const auto UPRD_MEMORY_PERCENTAGE = dmlc::GetEnv("UPRD_MEMORY_PERCENTAGE", 0.8);

static const auto UPR_INPUT_CHANNELS = dmlc::GetEnv("UPR_INPUT_CHANNELS", 3);
static const auto UPR_INPUT_WIDTH    = dmlc::GetEnv("UPR_INPUT_WIDTH", 224);
static const auto UPR_INPUT_HEIGHT   = dmlc::GetEnv("UPR_INPUT_HEIGHT", 224);

static const auto UPR_INPUT_MEAN_R = dmlc::GetEnv("UPR_INPUT_MEAN_R", 0);
static const auto UPR_INPUT_MEAN_G = dmlc::GetEnv("UPR_INPUT_MEAN_G", 0);
static const auto UPR_INPUT_MEAN_B = dmlc::GetEnv("UPR_INPUT_MEAN_B", 0);

static std::map<std::string, std::string> model_directory_paths{
    {"bvlc_alexnet_1.0", UPR_BASE_DIR + "bvlc_alexnet_1.0"},
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
    {"inceptionbn_21k_2.0", UPR_BASE_DIR + "inceptionbn_21k_2.0"},
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

static std::map<std::string, size_t> model_internal_memory_usage{{"bvlc_alexnet_1.0", 515367328},
                                                                 {"bvlc_googlenet_1.0", 111213336},
                                                                 {"bvlc_reference_caffenet_1.0", 511967136},
                                                                 {"bvlc_reference_rcnn_ilsvrc13_1.0", 485747928},
                                                                 {"dpn68_1.0", 121952160},
                                                                 {"dpn92_1.0", 340148136},
                                                                 {"inception_bn_3.0", 141964264},
                                                                 {"network_in_network_1.0", 131029280},
                                                                 {"o_resnet101_2.0", 427697664},
                                                                 {"o_resnet152_2.0", 553366032},
                                                                 {"o_vgg16_1.0", 1227778448},
                                                                 {"o_vgg19_1.0", 1270256016},
                                                                 {"resnet101_1.0", 422836416},
                                                                 {"resnet101_2.0", 427846016},
                                                                 {"resnet152_1.0", 548354240},
                                                                 {"resnet152_11k_1.0", 720908664},
                                                                 {"resnet152_2.0", 553363840},
                                                                 {"resnet18_2.0", 154382672},
                                                                 {"resnet200_2.0", 589410832},
                                                                 {"resnet269_2.0", 889219456},
                                                                 {"resnet34_2.0", 235307344},
                                                                 {"resnet50_1.0", 270482112},
                                                                 {"resnet50_2.0", 275493392},
                                                                 {"resnext101_1.0", 375935240},
                                                                 {"resnext101_32x4d_1.0", 377705976},
                                                                 {"resnext26_32x4d_1.0", 146605560},
                                                                 {"resnext50_1.0", 221516040},
                                                                 {"resnext50_32x4d_1.0", 223958520},
                                                                 {"squeezenet_1.0", 33859928},
                                                                 {"squeezenet_1.1", 27743352},
                                                                 {"vgg16_1.0", 1227778448},
                                                                 {"vgg16_sod_1.0", 1198280840},
                                                                 {"vgg16_sos_1.0", 1195166368},
                                                                 {"vgg19_1.0", 1270256016},
                                                                 {"wrn50_2.0", 758331776}};
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
static std::string span_category_close         = "close";
static std::string span_category_serialization = "serialization";
static std::string span_category_ipc           = "ipc";
static std::string span_category_grpc          = "grpc";
static std::string span_category_mxnet_init    = "init";
static std::string span_category_ignore        = "ignore";

using span_props = std::map<std::string, std::string>;

static inline engine::OprExecStat *start_span(const std::string &name, std::string category) {
#if MXNET_USE_PROFILER
  const auto ctx = get_ctx();
  auto opr_stat  = engine::Profiler::Get()->AddOprStat(ctx.dev_type, ctx.dev_id, name);
  uint64_t tid   = std::hash<std::thread::id>()(std::this_thread::get_id());
  engine::SetOprCategory(opr_stat, category);
  engine::SetOprStart(opr_stat);
  return opr_stat;
#else
  return nullptr;
#endif
}

static inline engine::OprExecStat *start_span(const std::string &name, std::string category, span_props props) {
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

#define SPAN_PRIVATE_UNIQUE_ID __LINE__

#define SPAN_PRIVATE_NAME SPAN_PRIVATE_CONCAT(__span__, SPAN_PRIVATE_UNIQUE_ID)
#define SPAN_PRIVATE_CONCAT(a, b) SPAN_PRIVATE_CONCAT2(a, b)
#define SPAN_PRIVATE_CONCAT2(a, b) a##b

#define TIME_IT(...)                                                                                                   \
  auto SPAN_PRIVATE_NAME = upr::start_span(                                                                            \
      #__VA_ARGS__, "statement",                                                                                       \
      span_props{{"function", __PRETTY_FUNCTION__}, {"file", __FILE__}, {"line", std::to_string(__LINE__)}});          \
  __VA_ARGS__;                                                                                                         \
  upr::stop_span(SPAN_PRIVATE_NAME);

static std::string get_model_name() {
  static const auto model_name = dmlc::GetEnv("UPR_MODEL_NAME", std::string(DEFAULT_MODEL));
  return model_name;
}

static size_t get_model_internal_memory_usage(std::string model_name = "") {
  if (model_name == "") {
    model_name = get_model_name();
  }
  const auto it = model_internal_memory_usage.find(model_name);
  if (it == model_internal_memory_usage.end()) {
    throw dmlc::Error(fmt::format("unable to find {} model in model_internal_memory_usage", model_name));
  }
  return it->second;
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

static size_t memory_free() {
  size_t device_free_physmem{0};
  size_t device_total_physmem{0};
  cudaMemGetInfo(&device_free_physmem, &device_total_physmem);
  return device_free_physmem;
}

static size_t memory_total() {
  size_t device_free_physmem{0};
  size_t device_total_physmem{0};
  cudaMemGetInfo(&device_free_physmem, &device_total_physmem);
  return device_total_physmem;
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

void Unload(mxnet::MXAPIPredictor *pred);

std::pair<std::string, std::string> Load(std::string model_name, std::vector<mxnet::NDArray> *data,
                                         std::vector<std::string> *keys);
} // namespace upr
#endif // MXNET_USE_CUDA
