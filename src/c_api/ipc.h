#pragma once

#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"

#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <dmlc/type_traits.h>
#include <iostream>
#include <map>
#include <memory>
#include <nnvm/node.h>
#include <string>
#include <vector>

namespace upr {

void UPRLoad(dmlc::Stream *fi, std::vector<NDArray> *data,
             std::vector<std::string> *keys);
}