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
 * Copyright (c) 2015 by Contributors
 * \file profiler.cc
 * \brief implements profiler
 */
#include "./profiler.h"
#include "../version.h"
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/omp.h>
#include <fstream>
#include <limits.h>
#include <mxnet/base.h>
#include <thread>
#include <unistd.h>

#include "c_api/ipc.h"
#include "json.hpp"

#if MXNET_USE_CUDA
#include "../common/cuda_utils.h"
#endif

#if defined(_MSC_VER) && _MSC_VER <= 1800
#include <Windows.h>
#endif

namespace mxnet {
namespace engine {

  using json = nlohmann::json;

  Profiler::Profiler() : state_(kNotRunning), status_(kNotStarted), enable_output_(false) {
    filename_        = dmlc::GetEnv("UPR_PROFILE_TARGET", std::string("profile.json"));
    this->init_time_ = NowInUsec();

    this->cpu_num_ = std::thread::hardware_concurrency();
#if MXNET_USE_CUDA
    int kMaxNumGpus = 32;
    this->gpu_num_  = kMaxNumGpus;
#else
    this->gpu_num_ = 0;
#endif

    this->profile_stat = new DevStat[cpu_num_ + gpu_num_ + 1];
    for (unsigned int i = 0; i < cpu_num_; ++i) {
      profile_stat[i].dev_id_   = i;
      profile_stat[i].dev_name_ = "cpu/" + std::to_string(i);
    }
    for (unsigned int i = 0; i < gpu_num_; ++i) {
      profile_stat[cpu_num_ + i].dev_name_ = "gpu/" + std::to_string(i);
    }
    profile_stat[cpu_num_ + gpu_num_].dev_name_ = "cpu pinned/";

    mode_ = (ProfilerMode) dmlc::GetEnv("MXNET_PROFILER_MODE", static_cast<int>(kAllOperator));
    if (dmlc::GetEnv("MXNET_PROFILER_AUTOSTART", 1)) {
      this->state_         = ProfilerState::kRunning;
      this->enable_output_ = true;
    }
  }

  Profiler *Profiler::Get() {
#if MXNET_USE_PROFILER
    static Profiler inst;
    return &inst;
#else
    return nullptr;
#endif
  }

  void Profiler::SetState(ProfilerState state) {
    std::lock_guard<std::mutex> lock{this->m_};
    // once running, output will be enabled.
    if (state == kRunning) {
      this->enable_output_ = true;
    }
    if (this->status_ == kNotStarted && state == kRunning) {
      this->init_time_ = NowInUsec();
      this->status_    = kStarted;
    }
    this->state_ = state;
  }

  void Profiler::SetConfig(ProfilerMode mode, std::string output_filename) {
    std::lock_guard<std::mutex> lock{this->m_};
    this->mode_     = mode;
    this->filename_ = output_filename;
  }

  OprExecStat *Profiler::AddOprStat(int dev_type, uint32_t dev_id) {
    return this->AddOprStat(dev_type, dev_id, "undefined");
  }
  OprExecStat *Profiler::AddOprStat(int dev_type, uint32_t dev_id, std::string opr_name) {
    std::unique_ptr<OprExecStat> opr_stat(new OprExecStat);
    opr_stat->category = "generic";
    opr_stat->dev_type = dev_type;
    opr_stat->dev_id   = dev_id;
    opr_stat->opr_name = opr_name;

    int idx;
    switch (dev_type) {
      case Context::kCPU:
        idx = dev_id;
        break;
      case Context::kGPU:
        idx = cpu_num_ + dev_id;
        break;
      case Context::kCPUPinned:
        idx = cpu_num_ + gpu_num_;
        break;
      default:
        LOG(FATAL) << "Unknown dev_type: " << dev_type;
        return NULL;
    }

    DevStat &dev_stat = profile_stat[idx];
    dev_stat.opr_exec_stats_->enqueue(opr_stat.get());
    return opr_stat.release();
  }

  static std::string _engine_type() {
    static const auto stype = dmlc::GetEnv("MXNET_ENGINE_TYPE", std::string("NaiveEngine"));
    return stype;
  }

  static std::string engine_type() {
    static auto engine = _engine_type();
    return engine;
  }

  static json emitPid(const DevStat &d) {
    const auto name = d.dev_name_;
    auto pid        = d.dev_id_;
    if (engine_type() == "NaiveEngine") {
      pid = 0;
    }
    json j = {{"ph", "M"},
              {"args", std::map<std::string, std::string>{{"name", name},
                                                          {"upr_enabled", upr::UPR_ENABLED ? "true" : "false"}}},
              {"pid", pid},
              {"name", "process_name"}};
    return j;
  }

  static std::string format_time(const std::time_t &r) {
    // static const int RFC3339NANO_SIZE = 36; /* 2006-01-02T15:04:05.999999999+00:00 */
    static const char *RFC3339Nano = "%04d-%02d-%02dT%02d:%02d:%02d.%09ld+00:00\n";
    const auto t                   = gmtime(&r);
    char tstamp[512];
    memset(tstamp, 0, sizeof tstamp);
    snprintf(tstamp, sizeof tstamp, RFC3339Nano, t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min,
             t->tm_sec, 0);
    auto str = std::string(tstamp);
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    return str;
  };

  static json emitEvent(const DevStat &d, const OprExecStat *opr_stat, std::string begin_end) {
    using namespace std::chrono;
    const auto name     = opr_stat->opr_name;
    const auto category = opr_stat->category;
    const auto ts       = begin_end == "B" ? opr_stat->opr_start_rel_micros : opr_stat->opr_end_rel_micros;
    auto pid            = d.dev_id_;
    auto tid            = opr_stat->thread_id;
    auto args           = opr_stat->metadata;

    args.insert({"upr_enabled", upr::UPR_ENABLED ? "true" : "false"});

    // if (engine_type() == "NaiveEngine") {
    pid = 0;
    tid = 0;
    //}
    // std::cout << "engine type = " << engine_type() << " \n";
    const auto init_time            = Profiler::Get()->GetInitTime();
    const auto duration_since_epoch = std::chrono::microseconds(init_time);
    const time_point<system_clock> tp_after_duration(duration_since_epoch);
    const time_t start_time = system_clock::to_time_t(tp_after_duration);
    json j                  = {{"name", name},
              {"cat", category},
              {"ph", begin_end},
              {"ts", ts},
              {"pid", pid},
              {"tid", tid},
              {"upr_enabled", upr::UPR_ENABLED},
              {"init_time", format_time(start_time)},
              {"args", args},
              {"start", opr_stat->opr_start_rel_micros},
              {"end", opr_stat->opr_end_rel_micros}};
    return j;
  }

  void Profiler::DumpProfile() {
    SetState(kNotRunning);

    std::lock_guard<std::mutex> lock{this->m_};
    json trace_events;

    uint32_t dev_num = cpu_num_ + gpu_num_ + 1;

    for (uint32_t i = 0; i < dev_num; ++i) {
      const DevStat &d = profile_stat[i];
      const auto pid   = emitPid(d);
      trace_events.emplace_back(pid);
    }

    for (uint32_t i = 0; i < dev_num; ++i) {
      DevStat &d = profile_stat[i];
      OprExecStat *_opr_stat;
      while (d.opr_exec_stats_->try_dequeue(_opr_stat)) {
        CHECK_NOTNULL(_opr_stat);
        auto opr_stat = _opr_stat;
        if (opr_stat == nullptr) {
          std::cout << "invalid oprstat";
          LOG(FATAL) << "invalid oprstat";
          continue;
        }
        const auto begin = emitEvent(d, opr_stat, "B");
        const auto end   = emitEvent(d, opr_stat, "E");
        trace_events.emplace_back(begin);
        trace_events.emplace_back(end);
        delete opr_stat;
      }
    }

    json metadata;

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 128
#endif
#ifndef LOGIN_NAME_MAX
#define LOGIN_NAME_MAX 128
#endif
    try {
      using namespace std::chrono;
      char hostname[HOST_NAME_MAX];
      char username[LOGIN_NAME_MAX];
      const std::time_t now           = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      const auto init_time            = Profiler::Get()->GetInitTime();
      const auto duration_since_epoch = std::chrono::microseconds(init_time);
      const time_point<system_clock> tp_after_duration(duration_since_epoch);
      const time_t start_time      = system_clock::to_time_t(tp_after_duration);
      const std::string run_id     = dmlc::GetEnv("UPR_RUN_ID", std::string("[undefined]"));
      const std::string git_sha    = dmlc::GetEnv("UPR_GIT_SHA", std::string(build_git_sha));
      const std::string git_branch = dmlc::GetEnv("UPR_GIT_BRANCH", std::string("[undefined]"));
      const std::string git_date   = dmlc::GetEnv("UPR_GIT_DATE", std::string(build_git_time));

      gethostname(hostname, HOST_NAME_MAX);
      getlogin_r(username, LOGIN_NAME_MAX);

      metadata =
          json({{"run_id", run_id},
                {"server",
                 {{"eviction_policy", upr::UPRD_EVICTION_POLICY},
                  {"estimation_rate", upr::UPRD_ESTIMATION_RATE},
                  {"memory_percentage", upr::UPRD_MEMORY_PERCENTAGE}}},
                {"hostname", std::string(hostname)},
                {"username", std::string(username)},
                {"git", {{"commit", git_sha}, {"date", git_date}, {"branch", git_branch}}},
                {"start_at", format_time(start_time)},
                {"end_at", format_time(now)},
                {"is_client", upr::is_client},
                {"upr_enabled", upr::UPR_ENABLED},
                {"upr_base_dir", upr::UPR_BASE_DIR},
                {"input",
                 {{"dimensions", json::array({upr::UPR_INPUT_CHANNELS, upr::UPR_INPUT_WIDTH, upr::UPR_INPUT_HEIGHT})},
                  {"mean", json::array({upr::UPR_INPUT_MEAN_R, upr::UPR_INPUT_MEAN_G, upr::UPR_INPUT_MEAN_B})}}},
                {"eager_mode", dmlc::GetEnv("UPR_INITIALIZE_EAGER", false)},
                {"eager_mode_async", dmlc::GetEnv("UPR_INITIALIZE_EAGER_ASYNC", false)},
                {"model_name", upr::get_model_name()},
                {"model_path", upr::get_model_directory_path()},
                {"model_params", upr::get_model_params_path()},
                {"symbol_params", upr::get_model_symbol_path()}});
    } catch (dmlc::Error &e) {
      metadata = json({{"error", e.what()}});
    } catch (const std::exception &e) {
      metadata = json({{"error", e.what()}});
    }

    enable_output_ = false;

    std::ofstream outfile(filename_);
    outfile //<< std::setw(4)
        << json{{"traceEvents", trace_events},
                {"upr_enabled", upr::UPR_ENABLED},
                {"displayTimeUnit", "ms"},
                {
                    "otherData",
                    metadata,
                }}
        << std::endl;
    outfile.flush();
    outfile.close();
  }

  inline uint64_t NowInUsec() {
#if defined(_MSC_VER) && _MSC_VER <= 1800
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return counter.QuadPart * 1000000 / frequency.QuadPart;
#else
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
#endif
  }

  void AddOprMetadata(OprExecStat *opr_stat, const std::string &key, const std::string &value) {
    opr_stat->metadata.insert({key, value});
  }

  void SetOprCategory(OprExecStat *opr_stat, const std::string &category) {
    opr_stat->category = category;
  }

  void SetOprStart(OprExecStat *opr_stat) {
    if (!opr_stat) {
      LOG(WARNING) << "SetOpStart: nullptr";
      return;
    }

#if MXNET_USE_CUDA
#if MXNET_USE_NVTX
    const auto name                   = opr_stat->opr_name;
    int color_id                      = opr_stat->category;
    color_id                          = color_id % num_colors;
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version               = NVTX_VERSION;
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType             = NVTX_COLOR_ARGB;
    eventAttrib.color                 = colors[color_id];
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii         = name;
    opr_stat->range_id                = nvtxRangeStartEx(&eventAttrib);
#endif
#endif

    opr_stat->opr_start_rel_micros = NowInUsec() - Profiler::Get()->GetInitTime();
  }

  void SetOprEnd(OprExecStat *opr_stat) {
#if MXNET_USE_CUDA
#if MXNET_USE_NVTX
    nvtxRangeEnd(opr_stat->range_id);
#endif
#endif
    opr_stat->opr_end_rel_micros = NowInUsec() - Profiler::Get()->GetInitTime();
  }

} // namespace engine
} // namespace mxnet
