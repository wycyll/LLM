# GEMM Migration Report: 20260524_run01

## Executive Summary

- Samples generated: 60
- Target-GPU compile attempted: 36/60 (60.0%)
- Compile success among attempted: 28/36 (77.8%)
- Aligned correctness shapes passed among attempted: 25/36 (69.4%)
- Performance measured among attempted: 15/36 (41.7%)
- Static WGMMA/TMA/cp.async claim without matching SASS confirmation among attempted samples: 13/36 (36.1%)

## Results By Task And Prompt

| task | prompt | samples | compile_attempted | compile | aligned_all_correct | perf_valid | target_static | target_sass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T10_h100_to_v100 | p0_no_hw_hint | 3 | 0/3 | not run | not run | not run | 33.3% | not run |
| T10_h100_to_v100 | p1_target_name_only | 3 | 0/3 | not run | not run | not run | 0.0% | not run |
| T10_h100_to_v100 | p2_hw_feature_table | 3 | 0/3 | not run | not run | not run | 0.0% | not run |
| T10_h100_to_v100 | p3_target_example | 3 | 0/3 | not run | not run | not run | 0.0% | not run |
| T1_v100_to_a100 | p0_no_hw_hint | 3 | 3/3 | 100.0% | 66.7% | 33.3% | 0.0% | 66.7% |
| T1_v100_to_a100 | p1_target_name_only | 3 | 3/3 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| T1_v100_to_a100 | p2_hw_feature_table | 3 | 3/3 | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| T1_v100_to_a100 | p3_target_example | 3 | 3/3 | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| T2_a100_to_h100 | p0_no_hw_hint | 3 | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| T2_a100_to_h100 | p1_target_name_only | 3 | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| T2_a100_to_h100 | p2_hw_feature_table | 3 | 3/3 | 33.3% | 33.3% | 33.3% | 66.7% | 0.0% |
| T2_a100_to_h100 | p3_target_example | 3 | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| T3_h100_to_a100 | p0_no_hw_hint | 3 | 3/3 | 66.7% | 66.7% | 66.7% | 0.0% | 66.7% |
| T3_h100_to_a100 | p1_target_name_only | 3 | 3/3 | 66.7% | 66.7% | 66.7% | 33.3% | 66.7% |
| T3_h100_to_a100 | p2_hw_feature_table | 3 | 3/3 | 66.7% | 0.0% | 0.0% | 66.7% | 66.7% |
| T3_h100_to_a100 | p3_target_example | 3 | 3/3 | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| T9_a100_to_v100 | p0_no_hw_hint | 3 | 0/3 | not run | not run | not run | 100.0% | not run |
| T9_a100_to_v100 | p1_target_name_only | 3 | 0/3 | not run | not run | not run | 100.0% | not run |
| T9_a100_to_v100 | p2_hw_feature_table | 3 | 0/3 | not run | not run | not run | 66.7% | not run |
| T9_a100_to_v100 | p3_target_example | 3 | 0/3 | not run | not run | not run | 33.3% | not run |

## Irregular-Shape Correctness Audit

This section uses `configs/shapes_correctness_audit.json` to stress non-tile-multiple boundary cases. It is stricter than the aligned correctness suite above.
Overall audit pass: 9/27 (33.3%).


| task | prompt | compiled_audited | audit_all_correct |
| --- | --- | ---: | ---: |
| T1_v100_to_a100 | p0_no_hw_hint | 3 | 1/3 (33.3%) |
| T1_v100_to_a100 | p1_target_name_only | 3 | 0/3 (0.0%) |
| T1_v100_to_a100 | p3_target_example | 3 | 0/3 (0.0%) |
| T2_a100_to_h100 | p0_no_hw_hint | 3 | 0/3 (0.0%) |
| T2_a100_to_h100 | p1_target_name_only | 3 | 0/3 (0.0%) |
| T2_a100_to_h100 | p2_hw_feature_table | 1 | 1/1 (100.0%) |
| T2_a100_to_h100 | p3_target_example | 3 | 3/3 (100.0%) |
| T3_h100_to_a100 | p0_no_hw_hint | 2 | 2/2 (100.0%) |
| T3_h100_to_a100 | p1_target_name_only | 2 | 2/2 (100.0%) |
| T3_h100_to_a100 | p2_hw_feature_table | 1 | 0/1 (0.0%) |
| T3_h100_to_a100 | p3_target_example | 3 | 0/3 (0.0%) |

## Baseline Prompt Ablation Findings

This section answers the baseline prompt question using only P0-P3. The stricter decision signal is the irregular-shape audit, not aligned-shape correctness alone.

| prompt | condition | compile | aligned_all_correct | irregular_audit | best_audit_pass_TFLOPS | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p0_no_hw_hint | no hardware hint | 8/9 | 7/8 | 3/8 | 18.039 | Some robust candidates exist, but success is target- and sample-dependent; no-hint migration is not reliable by itself. |
| p1_target_name_only | target name only | 8/9 | 8/8 | 2/8 | 3.703 | Target naming can help some samples, but aligned-suite success or TFLOPS must still be discounted when irregular audit fails. |
| p2_hw_feature_table | hardware feature table | 3/9 | 1/3 | 1/2 | 5.550 | The feature table can produce an occasional robust sample, but it also increases feature claims and compile instability. |
| p3_target_example | target example | 9/9 | 9/9 | 3/9 | 8.750 | Target examples are the most reliable baseline prompt overall, but passing examples still do not prove WGMMA/TMA-style target optimization. |

Answer: in this multi-target baseline, hardware hints are not a single monotonic improvement. No-hint and target-name prompts produce some audit-pass A100 samples, the feature table has only isolated success with compile instability, and target examples are strongest overall but not uniformly robust across target directions. Report the effect per task/prompt, and treat target examples as useful guidance rather than proof of hardware-optimal migration.

## Hardware Fit And Migration Boundary

The experiment distinguishes runnable GEMM migration from architecture-appropriate high-performance migration.

| hardware | GEMM style that fits | why it fits |
| --- | --- | --- |
| V100 / sm70 | WMMA/HMMA-style Tensor Core GEMM with cooperative global loads, shared-memory tiling, warp-level matrix fragments, and explicit boundary handling | Volta exposes Tensor Cores through WMMA/HMMA-era primitives but lacks `cp.async`, TMA, and WGMMA. The practical optimization target is good shared-memory reuse and coalesced loads rather than asynchronous copy engines. |
| A100 / sm80 | `mma.sync`/`ldmatrix` Tensor Core GEMM with `cp.async` multi-stage global-to-shared pipelines | Ampere adds `cp.async`, so a good A100 GEMM overlaps global memory copies with compute and feeds Tensor Cores through staged shared-memory tiles. This style is unavailable on V100 and still narrower than Hopper's warpgroup/TMA path. |
| H100 / sm90 | WGMMA/TMA/mbarrier/warpgroup-specialized GEMM, usually with producer-consumer scheduling | Hopper adds warpgroup matrix operations and Tensor Memory Accelerator bulk movement. A Hopper-style GEMM should reduce per-warp copy overhead, coordinate async tensor movement with barriers, and keep larger Tensor Core operations fed. |

The same code can become suboptimal or invalid after migration:

| source style moved to another GPU | expected issue |
| --- | --- |
| V100-style shared-memory/WMMA GEMM on A100/H100 | Usually compiles, but underuses Ampere/Hopper memory pipelines and newer Tensor Core issue paths. It may be correct but not a target-style migration. |
| A100 `cp.async` pipeline on V100 | `cp.async` is unsupported on sm70, so this is not portable without a separate fallback. |
| A100 `cp.async` pipeline on H100 | Often compiles and can run, but it misses H100-specific WGMMA/TMA/warpgroup mechanisms. It is an Ampere-style runnable kernel, not necessarily a Hopper-optimal kernel. |
| H100 WGMMA/TMA kernel on A100/V100 | WGMMA/TMA/mbarrier instructions and scheduling assumptions are sm90-specific, so code may fail compilation or require a separate architecture implementation. |
| Vectorized or `cp.async` copies across arbitrary shapes | 16-byte alignment and full-vector bounds must be proven. Several LLM kernels pass aligned shapes but fail irregular audit with misaligned-address or boundary errors. |

LLM capability boundary observed so far:

| capability | current evidence |
| --- | --- |
| Fixed ABI integration, simple shared-memory tiling, and conservative bounds checks | Achievable, especially when the prompt includes a target example. P3 is the most reliable baseline condition. |
| Producing target-feature vocabulary in source code | Easy for the model, especially with feature-table prompts, but weak evidence by itself. |
| Robust arbitrary-shape correctness | Possible for simpler shared-memory kernels, but not automatic. Irregular audit is required because aligned shapes hide boundary bugs. |
| Real H100 WGMMA/TMA/mbarrier/warpgroup implementation | Not demonstrated by P0-P3 baseline. Exploratory prompts increased feature claims but did not produce verified WGMMA/TMA SASS. |
| Performance-portable architecture selection | Not demonstrated. The model does not reliably choose the right hardware-specific strategy or tune tile sizes/stages/occupancy from hardware names or feature tables alone. |

## Run Scope And Caveats

- This run covers these prompt IDs only: `p0_no_hw_hint, p1_target_name_only, p2_hw_feature_table, p3_target_example`.
- The summary count `13/36` is a row-level rule: a sample is counted when `static_features` contains one of `wgmma, tma, cp_async` but `sass_features` contains none of those same tokens. It is not computed by subtracting aggregate table percentages, and one sample can contain multiple static features.
- `target_sass` is evidence that selected instruction-family tokens appear in SASS. It is not a performance claim and does not imply the kernel is correct, robust, or fast.
- Nsight Compute profile evidence is unavailable for this run, so hardware-feature confirmation here is limited to static source scanning and SASS inspection.
- Performance rows exist for 15 kernels, producing 90 shape-level rows. Interpret TFLOPS only after cross-checking the irregular-shape audit; aligned-correct but audit-failing kernels should not be reported as generally correct.

## Workflow Questions

### 1. Which hardware features matter for V100/A100/H100 GEMM?

- V100: WMMA or `mma.sync`, Volta Tensor Cores, shared-memory tiling, coalesced loads.
- A100: `cp.async`, multi-stage global-to-shared pipelines, `ldmatrix`, `mma.sync`, Ampere Tensor Cores.
- H100: TMA, WGMMA, `mbarrier`, warpgroup execution, producer-consumer warp specialization.

### 2. What happens without hardware hints?

Use the P0 row in the baseline ablation table above. In this multi-target baseline, no-hardware-hint generation can produce some audit-pass candidates, but success is task- and target-dependent and should not be treated as reliable hardware-style migration.

### 3. Do hardware hints and examples help?

In this multi-target baseline, hints and examples help unevenly. Target examples give the strongest overall robustness signal, especially for H100, but they are not uniformly correct across A100 back-migration tasks. Target-name and feature-table prompts can create some audit-pass samples, yet they also produce false positives and compile instability.

### 4. Do generated kernels compile, run, compute correctly, and perform?

The `compile_status`, `compile_success`, `correctness_status`, `performance_status`, `mean_tflops`, and `speedup_vs_cublas` columns in `migration_quality.csv` are the aligned-suite evidence. Use `correctness_audit_results.csv` when judging arbitrary-shape boundary correctness. Formal conclusions require target-GPU runs; samples with `compile_status=not_run` are generated but not yet evaluated on their target GPU.

### 5. Which features were only written but not verified?

Rows where `static_features` contains a target feature but `sass_features` and `profile_features` do not confirm it should be treated as `claimed_but_not_verified`. These cases are especially important for WGMMA/TMA/mbarrier claims.

## Notes

Type 6 and Type 7 require expert native-reference TFLOPS. Fill `native_ref_tflops` from validated V100/A100/H100 reference kernels before using those labels as final claims.
