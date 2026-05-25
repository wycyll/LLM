# GEMM Migration Report: 20260524_run01

## Executive Summary

- Samples generated: 60
- Target-GPU compile attempted: 12/60 (20.0%)
- Compile success among attempted: 10/12 (83.3%)
- Aligned correctness shapes passed among attempted: 10/12 (83.3%)
- Performance measured among attempted: 10/12 (83.3%)
- Static WGMMA/TMA/cp.async claim without matching SASS confirmation among attempted samples: 5/12 (41.7%)

## Results By Task And Prompt

| task | prompt | samples | compile_attempted | compile | aligned_all_correct | perf_valid | target_static | target_sass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T10_h100_to_v100 | p0_no_hw_hint | 3 | 0/3 | not run | not run | not run | 33.3% | not run |
| T10_h100_to_v100 | p1_target_name_only | 3 | 0/3 | not run | not run | not run | 0.0% | not run |
| T10_h100_to_v100 | p2_hw_feature_table | 3 | 0/3 | not run | not run | not run | 0.0% | not run |
| T10_h100_to_v100 | p3_target_example | 3 | 0/3 | not run | not run | not run | 0.0% | not run |
| T1_v100_to_a100 | p0_no_hw_hint | 3 | 0/3 | not run | not run | not run | 0.0% | not run |
| T1_v100_to_a100 | p1_target_name_only | 3 | 0/3 | not run | not run | not run | 0.0% | not run |
| T1_v100_to_a100 | p2_hw_feature_table | 3 | 0/3 | not run | not run | not run | 100.0% | not run |
| T1_v100_to_a100 | p3_target_example | 3 | 0/3 | not run | not run | not run | 100.0% | not run |
| T2_a100_to_h100 | p0_no_hw_hint | 3 | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| T2_a100_to_h100 | p1_target_name_only | 3 | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| T2_a100_to_h100 | p2_hw_feature_table | 3 | 3/3 | 33.3% | 33.3% | 33.3% | 66.7% | 0.0% |
| T2_a100_to_h100 | p3_target_example | 3 | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| T3_h100_to_a100 | p0_no_hw_hint | 3 | 0/3 | not run | not run | not run | 0.0% | not run |
| T3_h100_to_a100 | p1_target_name_only | 3 | 0/3 | not run | not run | not run | 33.3% | not run |
| T3_h100_to_a100 | p2_hw_feature_table | 3 | 0/3 | not run | not run | not run | 66.7% | not run |
| T3_h100_to_a100 | p3_target_example | 3 | 0/3 | not run | not run | not run | 100.0% | not run |
| T9_a100_to_v100 | p0_no_hw_hint | 3 | 0/3 | not run | not run | not run | 100.0% | not run |
| T9_a100_to_v100 | p1_target_name_only | 3 | 0/3 | not run | not run | not run | 100.0% | not run |
| T9_a100_to_v100 | p2_hw_feature_table | 3 | 0/3 | not run | not run | not run | 66.7% | not run |
| T9_a100_to_v100 | p3_target_example | 3 | 0/3 | not run | not run | not run | 33.3% | not run |

## Irregular-Shape Correctness Audit

This section uses `configs/shapes_correctness_audit.json` to stress non-tile-multiple boundary cases. It is stricter than the aligned correctness suite above.
Overall audit pass: 4/10 (40.0%).


| task | prompt | compiled_audited | audit_all_correct |
| --- | --- | ---: | ---: |
| T2_a100_to_h100 | p0_no_hw_hint | 3 | 0/3 (0.0%) |
| T2_a100_to_h100 | p1_target_name_only | 3 | 0/3 (0.0%) |
| T2_a100_to_h100 | p2_hw_feature_table | 1 | 1/1 (100.0%) |
| T2_a100_to_h100 | p3_target_example | 3 | 3/3 (100.0%) |

## Baseline Prompt Ablation Findings

This section answers the baseline prompt question using only P0-P3. The stricter decision signal is the irregular-shape audit, not aligned-shape correctness alone.

| prompt | condition | compile | aligned_all_correct | irregular_audit | best_audit_pass_TFLOPS | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p0_no_hw_hint | no hardware hint | 3/3 | 3/3 | 0/3 | - | No reliable migration signal: target-looking code may appear, but this run has no irregular-audit-pass P0 candidate. |
| p1_target_name_only | target name only | 3/3 | 3/3 | 0/3 | - | Target name alone is weak: aligned-suite success or TFLOPS must be discounted if irregular audit fails. |
| p2_hw_feature_table | hardware feature table | 1/3 | 1/1 | 1/1 | 5.550 | The feature table does not help by itself here; it increases feature claims but hurts compile stability. |
| p3_target_example | target example | 3/3 | 3/3 | 3/3 | 8.750 | The target example is the only robust baseline prompt in this run; it improves correctness, but does not prove Hopper WGMMA/TMA use. |

Answer: in this baseline, hardware hints help only when they include a target-style example. Hardware name alone and feature-table text mainly produce claims or aligned-suite false positives; they do not produce robust H100 GEMM migration evidence.

## Run Scope And Caveats

- This run covers these prompt IDs only: `p0_no_hw_hint, p1_target_name_only, p2_hw_feature_table, p3_target_example`.
- The summary count `5/12` is a row-level rule: a sample is counted when `static_features` contains one of `wgmma, tma, cp_async` but `sass_features` contains none of those same tokens. It is not computed by subtracting aggregate table percentages, and one sample can contain multiple static features.
- `target_sass` is evidence that selected instruction-family tokens appear in SASS. It is not a performance claim and does not imply the kernel is correct, robust, or fast.
- Nsight Compute profile evidence is unavailable for this run, so hardware-feature confirmation here is limited to static source scanning and SASS inspection.
- Performance rows exist for 10 kernels, producing 60 shape-level rows. Interpret TFLOPS only after cross-checking the irregular-shape audit; aligned-correct but audit-failing kernels should not be reported as generally correct.

## Workflow Questions

### 1. Which hardware features matter for V100/A100/H100 GEMM?

- V100: WMMA or `mma.sync`, Volta Tensor Cores, shared-memory tiling, coalesced loads.
- A100: `cp.async`, multi-stage global-to-shared pipelines, `ldmatrix`, `mma.sync`, Ampere Tensor Cores.
- H100: TMA, WGMMA, `mbarrier`, warpgroup execution, producer-consumer warp specialization.

### 2. What happens without hardware hints?

Use the P0 row in the baseline ablation table above. In this P0-P3 baseline, no-hardware-hint generation does not produce a robust irregular-audit-pass H100 migration.

### 3. Do hardware hints and examples help?

In this baseline, the target example P3 is the only prompt that reliably improves robustness. The target-name-only P1 and hardware-feature-table P2 variants mainly produce feature claims, compile failures, or aligned-suite false positives, not verified Hopper-style GEMM.

### 4. Do generated kernels compile, run, compute correctly, and perform?

The `compile_status`, `compile_success`, `correctness_status`, `performance_status`, `mean_tflops`, and `speedup_vs_cublas` columns in `migration_quality.csv` are the aligned-suite evidence. Use `correctness_audit_results.csv` when judging arbitrary-shape boundary correctness. Formal conclusions require target-GPU runs; samples with `compile_status=not_run` are generated but not yet evaluated on their target GPU.

### 5. Which features were only written but not verified?

Rows where `static_features` contains a target feature but `sass_features` and `profile_features` do not confirm it should be treated as `claimed_but_not_verified`. These cases are especially important for WGMMA/TMA/mbarrier claims.

## Notes

Type 6 and Type 7 require expert native-reference TFLOPS. Fill `native_ref_tflops` from validated V100/A100/H100 reference kernels before using those labels as final claims.
