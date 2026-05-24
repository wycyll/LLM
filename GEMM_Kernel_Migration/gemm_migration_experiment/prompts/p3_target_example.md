You are migrating a source GEMM kernel to a target NVIDIA GPU.

Source GPU: {source_gpu} / {source_arch}
Target GPU: {target_gpu} / {target_arch}

Target hardware notes:

{target_notes}

Target-style example snippet:

```cuda
{target_example}
```

Task:
Migrate the source GEMM kernel to the target GPU.
Preserve the GEMM semantics and adapt the code to the fixed interface below.

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
