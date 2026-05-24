#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__global__ void h100_fallback_tiled_gemm_kernel(const half* A, const half* B,
                                               half* C, int M, int N, int K) {
  __shared__ half tile_a[16][16];
  __shared__ half tile_b[16][16];

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int row = blockIdx.y * 16 + local_row;
  const int col = blockIdx.x * 16 + local_col;

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 16) {
    tile_a[local_row][local_col] =
        (row < M && k0 + local_col < K) ? A[row * K + k0 + local_col]
                                        : __float2half(0.0f);
    tile_b[local_row][local_col] =
        (k0 + local_row < K && col < N) ? B[(k0 + local_row) * N + col]
                                        : __float2half(0.0f);
    __syncthreads();

    if (row < M && col < N) {
      for (int kk = 0; kk < 16 && k0 + kk < K; ++kk) {
        acc += __half2float(tile_a[local_row][kk]) *
               __half2float(tile_b[kk][local_col]);
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
  h100_fallback_tiled_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
