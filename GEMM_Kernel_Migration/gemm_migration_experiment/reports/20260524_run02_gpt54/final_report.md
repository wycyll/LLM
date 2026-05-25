# GEMM Migration Report: 20260524_run02_gpt54

## Executive Summary

- Samples generated: 12
- Target-GPU compile attempted: 12/12 (100.0%)
- Compile success among attempted: 6/12 (50.0%)
- Aligned correctness shapes passed among attempted: 4/12 (33.3%)
- Performance measured among attempted: 4/12 (33.3%)
- Static WGMMA/TMA/cp.async claim without matching SASS confirmation among attempted samples: 5/12 (41.7%)

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

## Baseline Prompt Ablation Findings

This section answers the baseline prompt question using only P0-P3. The stricter decision signal is the irregular-shape audit, not aligned-shape correctness alone.

| prompt | condition | compile | aligned_all_correct | irregular_audit | best_audit_pass_TFLOPS | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p0_no_hw_hint | no hardware hint | 1/3 | 0/1 | 0/1 | - | No reliable migration signal: target-looking code may appear, but this run has no irregular-audit-pass P0 candidate. |
| p1_target_name_only | target name only | 2/3 | 1/2 | 0/2 | - | Target name alone is weak: aligned-suite success or TFLOPS must be discounted if irregular audit fails. |
| p2_hw_feature_table | hardware feature table | 0/3 | 0/3 | 0/0 | - | The feature table does not help by itself here; it increases feature claims but hurts compile stability. |
| p3_target_example | target example | 3/3 | 3/3 | 3/3 | 6.625 | The target example is the only robust baseline prompt in this run; it improves correctness, but does not prove Hopper WGMMA/TMA use. |

Answer: in this baseline, hardware hints help only when they include a target-style example. Hardware name alone and feature-table text mainly produce claims or aligned-suite false positives; they do not produce robust H100 GEMM migration evidence.

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
- The summary count `5/12` is a row-level rule: a sample is counted when `static_features` contains one of `wgmma, tma, cp_async` but `sass_features` contains none of those same tokens. It is not computed by subtracting aggregate table percentages, and one sample can contain multiple static features.
- `target_sass` is evidence that selected instruction-family tokens appear in SASS. It is not a performance claim and does not imply the kernel is correct, robust, or fast.
- Nsight Compute profile evidence is unavailable for this run, so hardware-feature confirmation here is limited to static source scanning and SASS inspection.
- Performance rows exist for 4 kernels, producing 24 shape-level rows. Interpret TFLOPS only after cross-checking the irregular-shape audit; aligned-correct but audit-failing kernels should not be reported as generally correct.

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
