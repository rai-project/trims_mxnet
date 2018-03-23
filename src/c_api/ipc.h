#pragma once

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

#define BYTE 1
#define KBYTE 1024 * BYTE
#define MBYTE 1024 * KBYTE
#define DATA_SIZE 4 * BYTE

#define DEFAULT_MODEL "alexnet"



using namespace mxnet;

static const auto HOME = dmlc::GetEnv("HOME", std::string("/home/abdul"));
static const auto is_client = dmlc::GetEnv("UPR_CLIENT", false);
static const auto CARML_HOME_BASE_DIR =
    HOME + std::string("/carml/data/mxnet/");
static const auto UPR_BASE_DIR = CARML_HOME_BASE_DIR;


static std::map<std::string, std::string> model_directory_paths{
    {"bvlc_alexnet_1.0", UPR_BASE_DIR + "bvlc_alexnet_1.0"},
    {"bvlc_googlenet_1.0", UPR_BASE_DIR + "bvlc_googlenet_1.0"},
    {"bvlc_reference_caffenet_1.0",
     UPR_BASE_DIR + "bvlc_reference_caffenet_1.0"},
    {"bvlc_reference_rcnn_ilsvrc13_1.0",
     UPR_BASE_DIR + "bvlc_reference_rcnn_ilsvrc13_1.0"},
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

static inline void force_runtime_initialization() {
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
    throw dmlc::Error(
        "unable to find the  model in model_direction_paths");
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

