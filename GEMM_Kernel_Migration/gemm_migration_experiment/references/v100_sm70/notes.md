# V100 / sm70 Notes

High-performance FP16 GEMM on V100 usually depends on Volta Tensor Cores through
WMMA or lower-level `mma.sync` instructions, careful shared-memory tiling, warp
mapping, and coalesced global memory movement.

V100 does not support `cp.async`, TMA, WGMMA, or Hopper `mbarrier` primitives.
Migration from V100 to A100 should preserve tiling/Tensor Core structure and may
introduce Ampere `cp.async` multi-stage copies. Migration from V100 to H100 may
need a deeper rewrite toward WGMMA/TMA if the target is Hopper-optimal.
