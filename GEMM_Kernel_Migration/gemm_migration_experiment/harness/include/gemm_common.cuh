#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct GemmShape {
  const char* shape_id;
  int M;
  int N;
  int K;
};

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                              \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status = (call);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,    \
                   static_cast<int>(status));                                \
      std::exit(3);                                                          \
    }                                                                        \
  } while (0)

extern "C" void launch_gemm_kernel(const half* A, const half* B, half* C,
                                    int M, int N, int K,
                                    cudaStream_t stream);

void run_cublas_reference(cublasHandle_t handle, const half* A, const half* B,
                          half* C, int M, int N, int K,
                          cudaStream_t stream);
