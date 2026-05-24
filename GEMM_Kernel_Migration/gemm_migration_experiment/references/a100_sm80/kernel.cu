#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ inline void cp_async_16B(void* shared_dst, const void* global_src,
                                    bool predicate) {
  const unsigned shared_addr = static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
  asm volatile(
      "{ .reg .pred p;"
      "  setp.ne.u32 p, %2, 0;"
      "  @p  cp.async.ca.shared.global [%0], [%1], 16;"
      "  @!p cp.async.ca.shared.global [%0], [%1], 16, 0;"
      "}\n"
      ::"r"(shared_addr), "l"(global_src), "r"(static_cast<unsigned>(predicate)));
}

__device__ inline void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

__global__ void a100_cp_async_scalar_gemm_kernel(const half* A, const half* B,
                                                 half* C, int M, int N, int K) {
  __shared__ __align__(16) half tile_a[16 * 16];
  __shared__ __align__(16) half tile_b[16 * 16];

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int row = blockIdx.y * 16 + local_row;
  const int col = blockIdx.x * 16 + local_col;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 16) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (tid < 32) {
      const int elem = tid * 8;
      const int a_row = blockIdx.y * 16 + elem / 16;
      const int a_col = k0 + elem % 16;
      const int b_row = k0 + elem / 16;
      const int b_col = blockIdx.x * 16 + elem % 16;
      const bool valid_a = a_row < M && a_col + 7 < K;
      const bool valid_b = b_row < K && b_col + 7 < N;
      cp_async_16B(tile_a + elem, A + a_row * K + a_col, valid_a);
      cp_async_16B(tile_b + elem, B + b_row * N + b_col, valid_b);
    }
    cp_async_commit();
    cp_async_wait();
#else
    if (tid < 256) {
      const int elem = tid;
      const int a_row = blockIdx.y * 16 + elem / 16;
      const int a_col = k0 + elem % 16;
      const int b_row = k0 + elem / 16;
      const int b_col = blockIdx.x * 16 + elem % 16;
      tile_a[elem] = (a_row < M && a_col < K) ? A[a_row * K + a_col]
                                              : __float2half(0.0f);
      tile_b[elem] = (b_row < K && b_col < N) ? B[b_row * N + b_col]
                                              : __float2half(0.0f);
    }
#endif
    __syncthreads();

    if (row < M && col < N) {
      for (int kk = 0; kk < 16 && k0 + kk < K; ++kk) {
        acc += __half2float(tile_a[local_row * 16 + kk]) *
               __half2float(tile_b[kk * 16 + local_col]);
      }
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = __float2half(acc);
  }
}

}  // namespace

extern "C" void launch_gemm_kernel(const half* A, const half* B, half* C,
                                    int M, int N, int K,
                                    cudaStream_t stream) {
  dim3 block(16, 16, 1);
  dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
  a100_cp_async_scalar_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
