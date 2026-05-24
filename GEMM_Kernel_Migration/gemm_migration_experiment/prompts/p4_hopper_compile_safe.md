You are migrating an FP16 GEMM kernel to NVIDIA H100 / Hopper / sm90.

The primary goal is a kernel that compiles with `nvcc -arch=sm_90`, runs without
illegal memory access, and passes all correctness shapes. Performance matters,
but do not sacrifice compileability or correctness.

Hard constraints:
- Return one complete CUDA source file only.
- Do not use CUTLASS, CuTe, CUB, Thrust, or external project headers.
- Do not use pseudo APIs or non-existent wrappers.
- Do not use `cuda::pipeline`, `cuda::barrier`, `cuda::memcpy_async`, TMA
  descriptors, or WGMMA unless you are certain the exact CUDA C++ or inline PTX
  syntax is valid for sm90.
- If you cannot write correct WGMMA/TMA code, produce a robust H100-compatible
  shared-memory or `cp.async` GEMM instead. This is better than fake Hopper code.
- Support arbitrary M, N, K from the test harness, including rectangular shapes.

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
