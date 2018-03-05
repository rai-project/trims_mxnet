/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 * \file initialize.cc
 * \brief initialize mxnet library
 */
#include "engine/profiler.h"
#include <dmlc/logging.h>
#include <mxnet/engine.h>
#include <signal.h>

#if DETAILED_PROFILE
#include "./inst_nvtx.inc"
#endif
#include "./version.inc"

// - apt-get install binutils-dev ...
// - g++/clang++ -lbfd ...
#define BACKWARD_HAS_BFD 1

#include "backward.hpp"

namespace backward {

backward::SignalHandling sh;

} // namespace backward

namespace mxnet {
#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE
static void SegfaultLogger(int sig) {
  using namespace backward;
  StackTrace st;
  st.load_here(32);
  std::cerr << "\nSegmentation fault: " << sig << "\n\n";
  Printer p;
  p.object     = true;
  p.color_mode = ColorMode::always;
  p.address    = true;
  p.print(st, stderr);
  exit(-1);
}
#endif

class LibraryInitializer {
public:
  LibraryInitializer() {
    dmlc::InitLogging("mxnet");
#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE
    signal(SIGSEGV, SegfaultLogger);
#endif
#if MXNET_USE_PROFILER
    // ensure profiler's constructor are called before atexit.
    engine::Profiler::Get();
    // DumpProfile will be called before engine's and profiler's destructor.
    std::atexit([]() {
      engine::Profiler* profiler = engine::Profiler::Get();
      if (profiler->IsEnableOutput()) {
        profiler->DumpProfile();
      }
    });
#endif

    static const auto eager_init       = dmlc::GetEnv("UPR_INTIALIZE_EAGER", false);
    static const auto eager_init_async = dmlc::GetEnv("UPR_INTIALIZE_EAGER_ASYNC", false);
    if (eager_init) {
      static const auto ctx = Context::GPU();
      auto engine           = Engine::_GetSharedRef();
      engine->Initalize(ctx);
    }
  }

  static LibraryInitializer* Get();
};

LibraryInitializer* LibraryInitializer::Get() {
  static LibraryInitializer inst;
  return &inst;
}

#ifdef __GNUC__
// Don't print an unused variable message since this is intentional
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

static LibraryInitializer* __library_init = LibraryInitializer::Get();
} // namespace mxnet
