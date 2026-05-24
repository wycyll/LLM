#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" void launch_gemm_kernel(const half* A, const half* B, half* C,
                                    int M, int N, int K,
                                    cudaStream_t stream);
