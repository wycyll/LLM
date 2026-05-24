# A100 / sm80 Notes

High-performance FP16 GEMM on A100 generally uses Ampere Tensor Cores,
`mma.sync`, `ldmatrix`, shared-memory tiling, and `cp.async` multi-stage global to
shared-memory pipelines.

Compared with V100, A100 adds asynchronous global-to-shared copies and deeper
software pipelines. Compared with H100, A100 does not support Hopper TMA or
WGMMA, so an H100-to-A100 migration must remove or replace Hopper-only features.
