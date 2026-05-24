You are given a GEMM kernel optimized for the source GPU.

Task:
Migrate this GEMM kernel to the target GPU.
Keep the GEMM semantics unchanged.
Make the migrated kernel correct and high-performance.

Source GPU: {source_gpu}
Target GPU: {target_gpu}

Required interface:

```cpp
extern "C" void launch_gemm_kernel(
    const half* A,
    const half* B,
    half* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
);
```

GEMM semantics:

```text
C[M, N] = A[M, K] x B[K, N]
A, B, C: half
layout: row-major
```

Return complete compilable CUDA code only. Do not rely on external project files.

Source kernel:

```cuda
{source_kernel}
```
