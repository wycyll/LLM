#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace {

__global__ void v100_wmma_gemm_kernel(const half* A, const half* B, half* C,
                                      int M, int N, int K) {
  __shared__ float c_tile[16 * 16];
  const int tile_m = blockIdx.y;
  const int tile_n = blockIdx.x;
  const int row = tile_m * 16;
  const int col = tile_n * 16;

  if (row >= M || col >= N) {
    return;
  }

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    const half* a_tile = A + row * K + k0;
    const half* b_tile = B + k0 * N + col;
    wmma::load_matrix_sync(a_frag, a_tile, K);
    wmma::load_matrix_sync(b_frag, b_tile, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);
  __syncthreads();

  for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
    const int local_row = idx / 16;
    const int local_col = idx % 16;
    if (row + local_row < M && col + local_col < N) {
      C[(row + local_row) * N + col + local_col] = __float2half(c_tile[idx]);
    }
  }
}

}  // namespace

extern "C" void launch_gemm_kernel(const half* A, const half* B, half* C,
                                    int M, int N, int K,
                                    cudaStream_t stream) {
  dim3 block(32, 1, 1);
  dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
  v100_wmma_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
