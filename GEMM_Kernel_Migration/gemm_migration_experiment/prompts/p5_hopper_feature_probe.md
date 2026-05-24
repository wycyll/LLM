You are migrating an A100-style FP16 GEMM kernel to NVIDIA H100 / Hopper / sm90.

This is a feature-probe experiment. Try to produce a real Hopper-style kernel,
not merely Ampere code compiled for sm90.

Target features to attempt:
- WGMMA / GMMA (`wgmma.mma_async` inline PTX or a valid CUDA abstraction)
- TMA / bulk tensor copy (`cp.async.bulk.tensor` or valid Tensor Map usage)
- warpgroup-level producer/consumer specialization
- mbarrier or equivalent Hopper synchronization

But the code must still be a complete CUDA source file that compiles with
`nvcc -arch=sm_90`. If you include a feature, use concrete code, not comments or
placeholder functions. Do not depend on CUTLASS, CuTe, or external headers.

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

Return complete compilable CUDA code only.

Source GPU: {source_gpu} / {source_arch}
Target GPU: {target_gpu} / {target_arch}

Source kernel:

```cuda
{source_kernel}
```
