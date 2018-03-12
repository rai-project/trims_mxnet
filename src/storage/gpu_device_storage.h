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
 * \file gpu_device_storage.h
 * \brief GPU storage implementation.
 */
#ifndef MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_GPU_DEVICE_STORAGE_H_

#include "../c_api/ipc.h"
#include "../common/cuda_utils.h"
#include "mxnet/base.h"
#include "mxnet/storage.h"
#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif // MXNET_USE_CUDA
#include <new>

#include "nvToolsExt.h"

#if MXNET_USE_NVTX
#ifndef PUSH_RANGE
#define PUSH_RANGE(name, cid)                                                                                          \
  {                                                                                                                    \
    int color_id                      = cid;                                                                           \
    static const uint32_t colors[]    = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,                               \
                                      0x0000ffff, 0x00ff0000, 0x00ffffff};                                          \
    const int num_colors              = sizeof(colors) / sizeof(uint32_t);                                             \
    color_id                          = color_id % num_colors;                                                         \
    nvtxEventAttributes_t eventAttrib = {0};                                                                           \
    eventAttrib.version               = NVTX_VERSION;                                                                  \
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                                 \
    eventAttrib.colorType             = NVTX_COLOR_ARGB;                                                               \
    eventAttrib.color                 = colors[color_id];                                                              \
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;                                                       \
    eventAttrib.message.ascii         = name;                                                                          \
    nvtxRangePushEx(&eventAttrib);                                                                                     \
  }
#define POP_RANGE() nvtxRangePop();
#endif
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE()
#endif

namespace mxnet {
namespace storage {
  /*!
   * \brief GPU storage implementation.
   */
  class GPUDeviceStorage {
  public:
    /*!
     * \brief Allocation.
     * \param size Size to allocate.
     * \return Pointer to the storage.
     */
    inline static void* Alloc(size_t size);
    /*!
     * \brief Deallocation.
     * \param ptr Pointer to deallocate.
     */
    inline static void Free(void* ptr);
  }; // class GPUDeviceStorage

  inline void* GPUDeviceStorage::Alloc(size_t size) {
    void* ret = nullptr;
#if MXNET_USE_CUDA
#if MXNET_USE_NCCL
    std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
#endif // MXNET_USE_NCCL
    cudaError_t e;

    if (upr::UPR_ENABLE_MEMORY_PROFILE) {
      auto span = upr::start_span("cudaMalloc", "memory", upr::span_props{{"size", std::to_string(size)}});
      e         = cudaMalloc(&ret, size);
      upr::stop_span(span);
    } else {
      e = cudaMalloc(&ret, size);
    }
    // LOG(INFO) << "allocating " << size << " bytes of memory using naive cuda storage. device_ptr = " << (size_t) ret;
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(ERROR) << "failed to perform cuda malloc " << cudaGetErrorString(e);
      throw std::bad_alloc();
    }
#else  // MXNET_USE_CUDA
    LOG(FATAL) << "Please compile with CUDA enabled";
#endif // MXNET_USE_CUDA
    return ret;
  }

  inline void GPUDeviceStorage::Free(void* ptr) {
#if MXNET_USE_CUDA
#if MXNET_USE_NCCL
    std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
#endif // MXNET_USE_NCCL
    // throw special exception for caller to catch.

    // LOG(INFO) << "freeing " << (size_t) ptr << " using naive cuda storage.";
    PUSH_RANGE("GPUDeviceStorage Free()", 1);

    cudaError_t err = cudaSuccess;
    if (upr::UPR_ENABLE_MEMORY_PROFILE) {
      auto span = upr::start_span("cudaFree", "memory");
      if (upr::UPR_ENABLE_CUDA_FREE) {
        err = cudaFree(ptr);
      }
      upr::stop_span(span);
    } else if (upr::UPR_ENABLE_CUDA_FREE) {
      err = cudaFree(ptr);
    }

    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
    POP_RANGE();

    LOG(FATAL) << "Please compile with CUDA enabled";
#endif // MXNET_USE_CUDA
  }

} // namespace storage
} // namespace mxnet

#endif // MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
