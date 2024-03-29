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
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif // MXNET_USE_CUDA
#include "../common/cuda_utils.h"
#include "./storage_manager.h"
#include <mutex>
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <new>
#include <unordered_map>
#include <vector>

#include "nvToolsExt.h"
#if MXNET_USE_NVTX
#ifndef PUSH_RANGE
#define PUSH_RANGE(name,cid) { \
                  int color_id = cid; \
          static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00,             0x00ff00ff,      0x0000ffff, 0x00ff0000, 0x00ffffff }; \                                     const int num_colors = sizeof(colors)/sizeof(uint32_t); \
                  color_id = color_id%num_colors;\
                  nvtxEventAttributes_t eventAttrib = {0}; \
                  eventAttrib.version = NVTX_VERSION; \
                  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
                  eventAttrib.colorType = NVTX_COLOR_ARGB; \
                  eventAttrib.color = colors[color_id]; \
                  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
                  eventAttrib.message.ascii = name; \
                  nvtxRangePushEx(&eventAttrib); \
          }
#define POP_RANGE() nvtxRangePop();
#endif 
#else
#define PUSH_RANGE(name, cid) 
#define POP_RANGE()
#endif


namespace mxnet {
namespace storage {

#if MXNET_USE_CUDA
/*!
 * \brief Storage manager with a memory pool on gpu.
 */
class GPUPooledStorageManager final : public StorageManager {
public:
  /*!
   * \brief Default constructor.
   */
  GPUPooledStorageManager() {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledStorageManager() { ReleaseAll(); }

  void Alloc(Storage::Handle *handle) override;
  void Free(Storage::Handle handle) override;

  void DirectFree(Storage::Handle handle) override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    DirectFreeNoLock(handle);
  }

 private:
  void DirectFreeNoLock(Storage::Handle handle) {

PUSH_RANGE("DirectFreeNoLock in pooled storage manager", 0);
      cudaError_t err = cudaFree(handle.dptr);
      size_t size = handle.size + NDEV;
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
POP_RANGE();
    used_memory_ -= size;
  }

private:
  void ReleaseAll();
  // used memory
  size_t used_memory_ = 0;
  // percentage of reserved memory
  int reserve_;
  // number of devices
  const int NDEV = 32;
  // memory pool
  std::unordered_map<size_t, std::vector<void *>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledStorageManager);
}; // class GPUPooledStorageManager

void GPUPooledStorageManager::Alloc(Storage::Handle *handle) {
  static const bool is_client = dmlc::GetEnv("UPR_CLINET", 0);
    LOG(INFO) << "requesting to allocate " << handle->size + NDEV
              << " bytes of memory on the gpu using a pooling allocator";
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size_t size = handle->size + NDEV;
  auto &&reuse_it = memory_pool_.find(size);
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free <= total * reserve_ / 100 || size > free - total * reserve_ / 100)
      ReleaseAll();

    void *ret = nullptr;
    cudaError_t e = cudaMalloc(&ret, size);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e)
                 << ". requesting to allocate " << size
                 << " bytes of gpu memory";
    }
    used_memory_ += size;
    handle->dptr = ret;
  } else {
    auto &&reuse_pool = reuse_it->second;
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    handle->dptr = ret;
  }
}

void GPUPooledStorageManager::Free(Storage::Handle handle) {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size_t size = handle.size + NDEV;
  auto &&reuse_pool = memory_pool_[size];
  reuse_pool.push_back(handle.dptr);
}

void GPUPooledStorageManager::ReleaseAll() {
  for (auto &&i : memory_pool_) {
    for (auto &&j : i.second) {
      Storage::Handle handle;
      handle.dptr = j;
      handle.size = i.first - NDEV;
      DirectFreeNoLock(handle);
    }
  }
  memory_pool_.clear();
}

#endif // MXNET_USE_CUDA

} // namespace storage
} // namespace mxnet

#endif // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
