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
 *  Copyright (c) 2015 by Contributors
 * \file naive_engine.cc
 * \brief Implementation of NaiveEngine
 */
#include "./engine_impl.h"
#include "../profiler/profiler.h"
#include "./openmp.h"
#include "c_api/ipc.h"
#include <atomic>
#include <thread>
#include <vector>
#include <future>

namespace mxnet {
namespace engine {

/* <<<<<<< HEAD */
/*   // implement naive engine */
/*   class NaiveEngine final : public Engine { */
/*   public: */
/*     struct NaiveOpr : public Opr { */
/*       AsyncFn fn; */
/*       std::vector<VarHandle> const_vars; */
/*       std::vector<VarHandle> mutable_vars; */
/*       FnProperty prop; */
/*       const char *opr_name; */
/*       /*! \brief indicate whether to profile this operator *1/ */
/*       bool profiling{false}; */
/*       /*! \brief operator execution statistics *1/ */
/*       OprExecStat *opr_stat; */
/*     }; */

/*     NaiveEngine() { */
/*     } */
/*     // virtual destructor */
/*     virtual ~NaiveEngine() { */
/*     } */

/*     virtual void InitializeAsync(const Context &exec_ctx) override { */
/*       init_future_ = std::async(std::launch::async, [this, &exec_ctx] { this->Initialize(exec_ctx); }); */
/*     } */

/*     virtual void Initialize(const Context &exec_ctx) override { */
/* ======= */
// implement naive engine
class NaiveEngine final : public Engine {
 public:
  struct NaiveOpr : public Opr {
    AsyncFn fn;
    std::vector<VarHandle> const_vars;
    std::vector<VarHandle> mutable_vars;
    FnProperty prop;
    const char* opr_name;
    /*! \brief indicate whether to profile this operator */
    bool profiling{false};
    /*! \brief operator execution statistics */
    std::unique_ptr<profiler::ProfileOperator> opr_profile;
  };

  NaiveEngine() {
  }
  // virtual destructor
  virtual ~NaiveEngine() {
#if MXNET_USE_CUDA
      using namespace upr;
      static bool initialized = false;
      if (initialized) {
        return;
      }
      int dev_id = static_cast<int>(exec_ctx.dev_id);
      MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(exec_ctx.dev_id));
      streams_.resize(dev_id + 1, nullptr);
      static const auto eager_init = dmlc::GetEnv("UPR_INITIALIZE_EAGER", false);
      if (!eager_init) {
        auto span        = start_span("performing NaiveEngine initialization", span_category_mxnet_init);
        streams_[dev_id] = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0, dev_id);
        stop_span(span);
      } else {
        streams_[dev_id] = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0, dev_id);
      }
      initialized = true;
#endif // MXNET_USE_CUDA
    }

    // new variables
    VarHandle NewVariable() override {
      size_t v = ++counter_;
      return reinterpret_cast<VarHandle>(v);
    }

  OprHandle NewOperator(AsyncFn fn,
                        std::vector<VarHandle> const& const_vars,
                        std::vector<VarHandle> const& mutable_vars,
                        FnProperty prop = FnProperty::kNormal,
                        const char* opr_name = nullptr,
                        bool wait = false) override {
    NaiveOpr *opr = new NaiveOpr();
    opr->fn = fn;
    opr->const_vars = const_vars;
    opr->mutable_vars = mutable_vars;
    opr->prop = prop;
    opr->opr_name = opr_name;
    return opr;
  }

    void DeleteOperator(OprHandle op) override {
      NaiveOpr *opr = op->Cast<NaiveOpr>();
      delete opr;
    }

  void Push(OprHandle op, Context exec_ctx, int priority = 0, bool profiling = false) override {
    profiler::Profiler *profiler = profiler::Profiler::Get();
    NaiveOpr *opr = op->Cast<NaiveOpr>();
    opr->profiling = profiling && profiler->IsProfiling(profiler::Profiler::kSymbolic);
    this->PushAsync([&](RunContext ctx, CallbackOnComplete on_complete) {
        if (opr->profiling) {
          std::unique_ptr<profiler::ProfileOperator::Attributes> attrs;
          if (profiler->AggregateEnabled()) {
            attrs.reset(new profiler::ProfileOperator::Attributes());
          }
          opr->opr_profile.reset(new profiler::ProfileOperator(opr->opr_name, attrs.release()));
          opr->opr_profile->start(exec_ctx.dev_type, exec_ctx.dev_id);
        }
        opr->fn(ctx, on_complete);
        if (opr->profiling) {
          opr->opr_profile->stop();
        }
      },
      exec_ctx,
      opr->const_vars,
      opr->mutable_vars,
      opr->prop,
      priority,
      opr->opr_name);
  }

  void PushAsync(AsyncFn exec_fun,
                 Context exec_ctx,
                 std::vector<VarHandle> const& const_vars,
                 std::vector<VarHandle> const& mutable_vars,
                 FnProperty prop = FnProperty::kNormal,
                 int priority = 0,
                 const char* opr_name = nullptr,
                 bool wait = false) override {
    CallbackOnComplete callback = CreateCallback(
        NaiveEngine::OnComplete, nullptr);
    this->req_completed_ = false;
    profiler::Profiler *profiler = profiler::Profiler::Get();
    NaiveOpr *opr = nullptr;
    const bool profiling = opr_name && profiler->IsProfiling(profiler::Profiler::kImperative);
    if (profiling) {
      opr = NewOperator(exec_fun, const_vars, mutable_vars,
                        prop, opr_name)->Cast<NaiveOpr>();
      opr->profiling = profiling;
      std::unique_ptr<profiler::ProfileOperator::Attributes> attrs;
      if (profiler->AggregateEnabled()) {
        attrs.reset(new profiler::ProfileOperator::Attributes());
      }
      opr->opr_profile.reset(new profiler::ProfileOperator(opr->opr_name, attrs.release()));
      opr->opr_profile->start(exec_ctx.dev_type, exec_ctx.dev_id);
    }
    if (exec_ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
        static const bool eager_init       = dmlc::GetEnv("UPR_INITIALIZE_EAGER", false);
        static const auto eager_init_async = dmlc::GetEnv("UPR_INITIALIZE_EAGER_ASYNC", false);
        if (eager_init_async) {
          this->init_future_.wait();
        } else if (!eager_init) {
          this->Initialize(exec_ctx);
        }
        size_t dev_id = static_cast<size_t>(exec_ctx.dev_id);
        exec_fun(RunContext{exec_ctx, streams_[dev_id]}, callback);
#else
        LOG(FATAL) << "GPU is not enabled";
#endif
    } else {
      exec_fun(RunContext{exec_ctx, &cpu_stream_}, callback);
    }
    CHECK(this->req_completed_)
        << "NaiveEngine only support synchronize Push so far";
    if (profiling) {
      opr->opr_profile->stop();
    }
  }

  void DeleteVariable(SyncFn delete_fn, Context exec_ctx, VarHandle var) override {
    this->PushSync(delete_fn, exec_ctx, {}, {var},
                   FnProperty::kNormal, 0, "DeleteVariable");
  }

    void WaitForVar(VarHandle var) override {
    }

    void WaitForAll() override {
    }

    void NotifyShutdown() override {
      shutdown_phase_.store(true);
    }

  private:
    // callback to oncomplete
    static void OnComplete(Engine *engine, void *param) {
      static_cast<NaiveEngine *>(engine)->req_completed_ = true;
    }
    // whether action is completed
    bool req_completed_;
    // counter
    std::atomic<size_t> counter_{0};
    /*! \brief whether it is during shutdown phase*/
    std::atomic<bool> shutdown_phase_{false};
    // CPU stream
    mshadow::Stream<cpu> cpu_stream_;
    // GPU streams
    std::vector<mshadow::Stream<gpu> *> streams_;
    // iniitalization future if eager async initialization is enabled
    std::future<void> init_future_;
  }; // class NaiveEngine

  Engine *CreateNaiveEngine() {
    return new NaiveEngine();
  }
  // whether action is completed
  bool req_completed_;
  // counter
  std::atomic<size_t> counter_{0};
  /*! \brief whether it is during shutdown phase*/
  std::atomic<bool> shutdown_phase_{false};
  // CPU stream
  mshadow::Stream<cpu> cpu_stream_;
  // GPU streams
  std::vector<mshadow::Stream<gpu>*> streams_;
};  // class NaiveEngine

Engine *CreateNaiveEngine() {
  return new NaiveEngine();
}
}  // namespace engine
}  // namespace mxnet
