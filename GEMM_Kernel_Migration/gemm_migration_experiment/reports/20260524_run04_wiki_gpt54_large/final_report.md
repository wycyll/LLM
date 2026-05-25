# GEMM Migration Report: 20260524_run04_wiki_gpt54_large

## Executive Summary

- Samples generated: 40
- Target-GPU compile attempted: 40/40 (100.0%)
- Compile success among attempted: 35/40 (87.5%)
- Aligned correctness shapes passed among attempted: 31/40 (77.5%)
- Performance measured among attempted: 20/40 (50.0%)
- Static WGMMA/TMA/cp.async claim without matching SASS confirmation among attempted samples: 23/40 (57.5%)

## Results By Task And Prompt

| task | prompt | samples | compile_attempted | compile | aligned_all_correct | perf_valid | target_static | target_sass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2_a100_to_h100 | p3_target_example | 10 | 10/10 | 100.0% | 100.0% | 90.0% | 10.0% | 0.0% |
| T2_a100_to_h100 | p4_hopper_compile_safe | 10 | 10/10 | 100.0% | 90.0% | 0.0% | 100.0% | 30.0% |
| T2_a100_to_h100 | p6_kernelwiki_hopper_audit_safe | 10 | 10/10 | 80.0% | 60.0% | 60.0% | 100.0% | 20.0% |
| T2_a100_to_h100 | p7_kernelwiki_hopper_feature_evidence | 10 | 10/10 | 70.0% | 60.0% | 50.0% | 100.0% | 70.0% |

## Irregular-Shape Correctness Audit

This section uses `configs/shapes_correctness_audit.json` to stress non-tile-multiple boundary cases. It is stricter than the aligned correctness suite above.
Overall audit pass: 20/33 (60.6%).


| task | prompt | compiled_audited | audit_all_correct |
| --- | --- | ---: | ---: |
| T2_a100_to_h100 | p3_target_example | 10 | 9/10 (90.0%) |
| T2_a100_to_h100 | p4_hopper_compile_safe | 10 | 0/10 (0.0%) |
| T2_a100_to_h100 | p6_kernelwiki_hopper_audit_safe | 7 | 6/7 (85.7%) |
| T2_a100_to_h100 | p7_kernelwiki_hopper_feature_evidence | 6 | 5/6 (83.3%) |

## Run Scope And Caveats

- This run covers these prompt IDs only: `p3_target_example, p4_hopper_compile_safe, p6_kernelwiki_hopper_audit_safe, p7_kernelwiki_hopper_feature_evidence`.
- It does not include the baseline prompts `p0_no_hw_hint`, `p1_target_name_only`, `p2_hw_feature_table`. Use P0/P1/P2-containing runs for no-hardware-hint, target-name-only, and hardware-feature-table conclusions.
- The summary count `23/40` is a row-level rule: a sample is counted when `static_features` contains one of `wgmma, tma, cp_async` but `sass_features` contains none of those same tokens. It is not computed by subtracting aggregate table percentages, and one sample can contain multiple static features.
- `target_sass` is evidence that selected instruction-family tokens appear in SASS. It is not a performance claim and does not imply the kernel is correct, robust, or fast.
- Nsight Compute profile evidence is unavailable for this run, so hardware-feature confirmation here is limited to static source scanning and SASS inspection.
- Performance was measured for 20 kernels selected after the irregular-shape audit, producing 120 shape-level rows. Treat `perf_valid` as an audit-pass selected subset, not as a measurement over every aligned-correct kernel.
- `p4_hopper_compile_safe` has no performance rows because it passed 0/10 irregular-audit cases and was filtered out before formal timing. The audit logs include `misaligned address` failures. Do not interpret its `perf_valid=0` as a measured slow kernel.

## Workflow Questions

### 1. Which hardware features matter for V100/A100/H100 GEMM?

- V100: WMMA or `mma.sync`, Volta Tensor Cores, shared-memory tiling, coalesced loads.
- A100: `cp.async`, multi-stage global-to-shared pipelines, `ldmatrix`, `mma.sync`, Ampere Tensor Cores.
- H100: TMA, WGMMA, `mbarrier`, warpgroup execution, producer-consumer warp specialization.

### 2. What happens without hardware hints?

This run does not include `p0_no_hw_hint`, so it cannot answer the no-hardware-hint question by itself. Use the P0-containing baseline runs for that comparison; this run is scoped to target-example, compile-safe, and KernelWiki-informed H100 prompts.

### 3. Do hardware hints and examples help?

This run compares the available prompt variants in the table above. For the original P0/P1/P2/P3 hint ablation, use a run that contains all four baseline prompts; run04 mainly tests whether compile-safe and KernelWiki-informed prompts improve robustness or feature evidence relative to P3/P4-style prompts.

### 4. Do generated kernels compile, run, compute correctly, and perform?

The `compile_status`, `compile_success`, `correctness_status`, `performance_status`, `mean_tflops`, and `speedup_vs_cublas` columns in `migration_quality.csv` are the aligned-suite evidence. Use `correctness_audit_results.csv` when judging arbitrary-shape boundary correctness. Formal conclusions require target-GPU runs; samples with `compile_status=not_run` are generated but not yet evaluated on their target GPU.

### 5. Which features were only written but not verified?

Rows where `static_features` contains a target feature but `sass_features` and `profile_features` do not confirm it should be treated as `claimed_but_not_verified`. These cases are especially important for WGMMA/TMA/mbarrier claims.

## Notes

Type 6 and Type 7 require expert native-reference TFLOPS. Fill `native_ref_tflops` from validated V100/A100/H100 reference kernels before using those labels as final claims.
