# GEMM Migration Report: 20260524_run02_gpt54

## Executive Summary

- Samples generated: 12
- Target-GPU compile attempted: 12/12 (100.0%)
- Compile success among attempted: 6/12 (50.0%)
- Aligned correctness shapes passed among attempted: 4/12 (33.3%)
- Performance measured among attempted: 4/12 (33.3%)
- Static target-feature claim without SASS confirmation among attempted samples: 5/12 (41.7%)

## Results By Task And Prompt

| task | prompt | samples | compile_attempted | compile | aligned_all_correct | perf_valid | target_static | target_sass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2_a100_to_h100 | p0_no_hw_hint | 3 | 3/3 | 33.3% | 0.0% | 0.0% | 100.0% | 33.3% |
| T2_a100_to_h100 | p1_target_name_only | 3 | 3/3 | 66.7% | 33.3% | 33.3% | 33.3% | 33.3% |
| T2_a100_to_h100 | p2_hw_feature_table | 3 | 3/3 | 0.0% | 0.0% | 0.0% | 33.3% | 0.0% |
| T2_a100_to_h100 | p3_target_example | 3 | 3/3 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |

## Irregular-Shape Correctness Audit

This section uses `configs/shapes_correctness_audit.json` to stress non-tile-multiple boundary cases. It is stricter than the aligned correctness suite above.
Overall audit pass: 3/6 (50.0%).


| task | prompt | compiled_audited | audit_all_correct |
| --- | --- | ---: | ---: |
| T2_a100_to_h100 | p0_no_hw_hint | 1 | 0/1 (0.0%) |
| T2_a100_to_h100 | p1_target_name_only | 2 | 0/2 (0.0%) |
| T2_a100_to_h100 | p3_target_example | 3 | 3/3 (100.0%) |

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

The `compile_status`, `compile_success`, `correctness_status`, `performance_status`, `mean_tflops`, and `speedup_vs_cublas` columns in `migration_quality.csv` are the aligned-suite evidence. Use `correctness_audit_results.csv` when judging arbitrary-shape boundary correctness. Formal conclusions require target-GPU runs; samples with `compile_status=not_run` are generated but not yet evaluated on their target GPU.

### 5. Which features were only written but not verified?

Rows where `static_features` contains a target feature but `sass_features` and `profile_features` do not confirm it should be treated as `claimed_but_not_verified`. These cases are especially important for WGMMA/TMA/mbarrier claims.

## Notes

Type 6 and Type 7 require expert native-reference TFLOPS. Fill `native_ref_tflops` from validated V100/A100/H100 reference kernels before using those labels as final claims.
