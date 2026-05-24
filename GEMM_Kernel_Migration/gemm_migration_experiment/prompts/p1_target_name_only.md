Source GPU: NVIDIA {source_gpu}, {source_arch}.
Target GPU: NVIDIA {target_gpu}, {target_arch}.

Please migrate the GEMM kernel to the target GPU.
The output must preserve the original GEMM semantics and should be optimized for
the target GPU.

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

Return complete compilable CUDA code only. Do not rely on external project files.

Source kernel:

```cuda
{source_kernel}
```
