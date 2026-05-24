#include <cublas_v2.h>

#include "gemm_common.cuh"

void run_cublas_reference(cublasHandle_t handle, const half* A, const half* B,
                          half* C, int M, int N, int K,
                          cudaStream_t stream) {
  CHECK_CUBLAS(cublasSetStream(handle, stream));
  const float alpha = 1.0f;
  const float beta = 0.0f;

  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C,
                            CUDA_R_16F, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
