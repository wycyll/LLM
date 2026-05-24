You are migrating an A100-style FP16 GEMM kernel to NVIDIA H100 / Hopper / sm90.

This is a KernelWiki-informed feature-evidence experiment. The goal is to test
whether you can produce real Hopper code whose features survive compile and SASS
inspection, while still computing correctly for arbitrary row-major shapes.

KernelWiki guidance for H100/Hopper:
- Hopper GEMM features include GMMA/WGMMA, TMA, mbarrier, warpgroup execution,
  and warp-specialized producer/consumer pipelines.
- A Hopper WGMMA path uses warpgroup-scoped MMA and register accumulators.
- TMA is bulk tensor movement between global memory and shared memory and is
  typically paired with mbarrier/pipeline synchronization.
- Blackwell is different: `tcgen05`, TMEM accumulator storage, SM100 schedules,
  and Blackwell CuTe helpers are not valid H100 targets. Do not use them.

Feature attempt rules:
- You may attempt WGMMA/TMA/mbarrier only with concrete valid sm90 CUDA/PTX.
- If the exact syntax is uncertain, write an alignment-safe shared-memory or
  `cp.async` kernel and do not claim WGMMA/TMA in code names or comments.
- If using `cp.async`, handle odd K/N row strides safely. A 16-byte async copy
  requires 16-byte alignment and full 8-half in-bounds coverage; otherwise use
  scalar guarded loads for that region.
- The kernel must pass irregular shapes such as 127x129x131, 255x257x259,
  511x521x333, and 1000x777x999.

Hard constraints:
- Return one complete CUDA source file only.
- Do not use CUTLASS, CuTe, CUB, Thrust, or external project headers.
- Do not use Blackwell-only `tcgen05`, TMEM, or SM100 code.
- Do not use pseudo APIs or placeholders.
- Compile target is `nvcc -arch=sm_90`.

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

GEMM semantics: row-major `C[M,N] = A[M,K] x B[K,N]`, FP16 inputs/output,
FP32 accumulation.

Return complete compilable CUDA code only.

Source GPU: {source_gpu} / {source_arch}
Target GPU: {target_gpu} / {target_arch}

Source kernel:

```cuda
{source_kernel}
```
