You are migrating an FP16 GEMM kernel to NVIDIA H100 / Hopper / sm90.

Use these KernelWiki-derived facts as design guidance:
- Hopper high-performance GEMM normally relies on GMMA/WGMMA, TMA, mbarrier,
  warpgroup execution, and warp-specialized producer/consumer pipelines.
- Blackwell SM100 features are not Hopper features. Do not use `tcgen05`, TMEM,
  SM100 schedules, SM100 CuTe helpers, or Blackwell-only APIs for this H100
  target.
- TMA/WGMMA code is valuable only if it is concrete, compilable sm90 code. A
  source comment or placeholder name is not evidence.

This experiment stresses correctness on arbitrary M, N, K, including odd sizes
such as 127x129x131 and 1000x777x999. The generated kernel must pass those
irregular shapes, not only multiples of 16 or 128.

Hard correctness constraints:
- Return one complete CUDA source file only.
- Use the exact ABI below.
- Implement row-major `C[M,N] = A[M,K] x B[K,N]`.
- Use FP32 accumulation and write FP16 output.
- Every global load and store must be bounds checked.
- If using 16-byte `cp.async`, prove both source and destination are 16-byte
  aligned and the 8-half vector is fully in bounds. Otherwise use scalar guarded
  loads or a safe fallback path for that tile.
- Do not read from uninitialized shared memory on boundary tiles.
- Do not depend on CUTLASS, CuTe, CUB, Thrust, or external headers.
- Do not use pseudo APIs such as made-up `cuda::pipeline`, invalid
  `cuda::memcpy_async`, or incomplete TMA descriptor code.

Optimization preference:
1. Correct arbitrary-shape H100 code.
2. If you can do so with valid sm90 syntax, use real Hopper features.
3. If not, prefer a robust shared-memory or alignment-safe `cp.async` kernel
   over fake WGMMA/TMA code.

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

Return complete compilable CUDA code only.

Source GPU: {source_gpu} / {source_arch}
Target GPU: {target_gpu} / {target_arch}

Source kernel:

```cuda
{source_kernel}
```
