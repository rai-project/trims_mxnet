#include "./base64.h"
#include <cstring>
#include <stdexcept>

#include <cuda_runtime_api.h>

#include "mxnet/c_api.h"
#include "mxnet/c_predict_api.h"

#define CUDA_CHECK_CALL(func, msg)                                             \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "CUDA[" << msg << "]:: " << cudaGetErrorString(e) << "\n";  \
      throw std::runtime_error(cudaGetErrorString(e));                         \
    }                                                                          \
  }

int main(int argc, char **argv) {
  cudaFree(0);

  const std::string filename{"profile"};
  MXSetProfilerConfig(1, filename.c_str());
  // Stop profiling
  MXSetProfilerState(1);

  int version = 0;
  const auto err = MXGetVersion(&version);
  if (err) {
    std::cerr << "error :: " << err << " while getting mxnet version\n";
  }

  const std::string base64_handle = argv[1];
  const auto ipc_handle = base64_decode(base64_handle);
  cudaIpcMemHandle_t handle;
  memcpy((uint8_t *)&handle, ipc_handle.c_str(), sizeof(handle));
  float *data;
  CUDA_CHECK_CALL(cudaIpcOpenMemHandle((void **)&data, handle,
                                       cudaIpcMemLazyEnablePeerAccess),
                  "open");
  CUDA_CHECK_CALL(cudaIpcCloseMemHandle(data), "close");
  // Stop profiling
  MXSetProfilerState(0);
}
