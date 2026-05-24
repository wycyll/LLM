Hardware information:

V100 / sm70:
- Volta Tensor Core
- WMMA / mma.sync style matrix multiply
- no cp.async
- no TMA
- no WGMMA

A100 / sm80:
- Ampere Tensor Core
- supports cp.async
- supports multi-stage global-to-shared memory pipeline
- uses mma.sync style Tensor Core instructions
- no TMA
- no WGMMA

H100 / sm90:
- Hopper Tensor Core
- supports TMA
- supports WGMMA
- supports warpgroup-level execution
- supports mbarrier
- benefits from producer-consumer warp specialization

Task:
Migrate the source GEMM kernel to the target GPU.
Use target-specific hardware features when appropriate.

Source GPU: {source_gpu} / {source_arch}
Target GPU: {target_gpu} / {target_arch}

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
