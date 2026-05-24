# GEMM Migration Report: smoke_local_v3

## Executive Summary

- Samples: 1
- Compile success: 1/1 (100.0%)
- All correctness shapes passed: 1/1 (100.0%)
- Performance measured: 1/1 (100.0%)
- Static target-feature claim without SASS confirmation: 0/1 (0.0%)

## Results By Task And Prompt

| task | prompt | samples | compile | all_correct | perf_valid | target_static | target_sass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_reference | p0_no_hw_hint | 1 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |

## Workflow Questions

### 1. Which hardware features matter for V100/A100/H100 GEMM?

- V100: WMMA or `mma.sync`, Volta Tensor Cores, shared-memory tiling, coalesced loads.
- A100: `cp.async`, multi-stage global-to-shared pipelines, `ldmatrix`, `mma.sync`, Ampere Tensor Cores.
- H100: TMA, WGMMA, `mbarrier`, warpgroup execution, producer-consumer warp specialization.

### 2. What happens without hardware hints?

Use rows with `prompt_id=p0_no_hw_hint` to compare compile/correctness rates and whether static/SASS target features appear. If P0 compiles but has no H100 WGMMA/TMA evidence, it is a runnable migration rather than a Hopper-style migration.

### 3. Do hardware hints and examples help?

Compare P1, P2, and P3 against P0 in the table above. A useful hint should improve compile/correctness/performance or increase confirmed target-feature evidence, not only increase keyword frequency.

### 4. Do generated kernels compile, run, compute correctly, and perform?

The `compile_success`, `correctness_status`, `performance_status`, `mean_tflops`, and `speedup_vs_cublas` columns in `migration_quality.csv` are the primary evidence. Formal conclusions require target-GPU runs.

### 5. Which features were only written but not verified?

Rows where `static_features` contains a target feature but `sass_features` and `profile_features` do not confirm it should be treated as `claimed_but_not_verified`. These cases are especially important for WGMMA/TMA/mbarrier claims.

## Notes

Type 6 and Type 7 require expert native-reference TFLOPS. Fill `native_ref_tflops` from validated V100/A100/H100 reference kernels before using those labels as final claims.
