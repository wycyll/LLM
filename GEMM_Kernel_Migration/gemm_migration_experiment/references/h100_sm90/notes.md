# H100 / sm90 Notes

High-performance FP16 GEMM on H100 should be evaluated for Hopper-specific
features: WGMMA, TMA, `mbarrier`, warpgroup-level execution, producer-consumer
warp specialization, and tile shapes chosen for Hopper Tensor Cores.

An A100-to-H100 migration that only preserves `cp.async` and `mma.sync` may be
correct on H100, but it is Ampere-style rather than Hopper-optimal. A real
Hopper-style success needs static and preferably SASS/profile evidence for WGMMA
or TMA, plus correctness and meaningful performance on H100.
