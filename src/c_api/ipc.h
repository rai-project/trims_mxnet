#pragma once

#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <dmlc/omp.h>
#include <dmlc/recordio.h>
#include <dmlc/type_traits.h>
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
#include <nnvm/node.h>
#include <string>
#include <vector>

#define BYTE 1
#define KBYTE 1024 * BYTE
#define MBYTE 1024 * KBYTE
#define DATA_SIZE 4 * BYTE

namespace upr {

struct server {
  static std::string host_name;
  static int port;
  static std::string address;
};

void Load(std::string model_name, 
             std::vector<mxnet::NDArray> *data, std::vector<std::string> *keys);
}
