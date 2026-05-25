# T2 A100 to H100 Baseline Prompt Ablation

Scope: H100 target evaluation for `T2_a100_to_h100`, using at most four baseline prompts.

Active prompt set:

| prompt | condition | question |
| --- | --- | --- |
| `p0_no_hw_hint` | no hardware hint | Can the model infer a Hopper-appropriate GEMM migration by itself? |
| `p1_target_name_only` | target GPU name and SM only | Does naming H100/sm90 help? |
| `p2_hw_feature_table` | explicit V100/A100/H100 feature table | Do hardware facts become valid code? |
| `p3_target_example` | target notes and target-style example | Does example-based guidance help? |

Primary evidence uses `gpt-5.4 / 20260524_run02_gpt54`, because `azure/gpt-5.4` is the newest available deployment in this environment. The stricter correctness signal is the irregular audit in `configs/shapes_correctness_audit.json`; aligned correctness alone is not enough.

## Primary Baseline Results

| prompt | compile | aligned all-correct | irregular audit pass | measured TFLOPS | audit-pass best TFLOPS | SASS evidence | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `p0_no_hw_hint` | 1/3 | 0/1 | 0/1 | - | - | `mma` in 1 compiled sample | No reliable migration. The model writes target-looking code, but no P0 sample passes the stricter audit. |
| `p1_target_name_only` | 2/3 | 1/2 | 0/2 | 12.137 | - | `mma` in 1 compiled sample | H100/sm90 naming can produce an aligned-suite performance number, but it fails irregular audit and should not be reported as generally correct. |
| `p2_hw_feature_table` | 0/3 | 0/0 | 0/0 | - | - | none | Explicit hardware facts hurt compile stability in this run. Feature-table prompting increases claims, not verified kernels. |
| `p3_target_example` | 3/3 | 3/3 | 3/3 | 6.625 | 6.625 | none | Target examples help robustness. This is the only P0-P3 condition with complete compile/correctness/audit evidence. |

## Cross-Model Check

`gpt-5.2-chat / 20260524_run01` shows the same main pattern, even though compile rates differ:

| prompt | compile | aligned all-correct | irregular audit pass | audit-pass best TFLOPS | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `p0_no_hw_hint` | 3/3 | 3/3 | 0/3 | - | Aligned-suite false positives; generated `cp.async` code fails irregular alignment/bounds cases. |
| `p1_target_name_only` | 3/3 | 3/3 | 0/3 | - | Same failure mode as P0. |
| `p2_hw_feature_table` | 1/3 | 1/1 | 1/1 | 5.550 | One robust sample, but low compile stability and no WGMMA/TMA SASS evidence. |
| `p3_target_example` | 3/3 | 3/3 | 3/3 | 8.750 | Most reliable baseline prompt across both models. |

## Answers

1. No hardware hint does not reliably migrate A100 GEMM to H100. P0 can produce compilable or target-looking code, but it has no irregular-audit-pass candidate in either baseline run.
2. Target name alone is not enough. P1 may improve surface-level target awareness, but the best-looking `gpt-5.4` P1 sample fails irregular audit.
3. The hardware feature table does not reliably help. It increases textual feature claims and can reduce compile stability; it does not produce verified WGMMA/TMA SASS evidence.
4. The target example is useful. P3 is the only baseline prompt that consistently passes compile, aligned correctness, and irregular-shape audit across both models.
5. None of the P0-P3 baseline runs produce confirmed H100 WGMMA or TMA SASS. The successful P3 kernels should be described as robust runnable GEMM migrations, not Hopper-style high-performance WGMMA/TMA migrations.

## Reporting Guidance

- Use P0-P3 as the formal prompt ablation. Do not add more prompt IDs to the main experiment unless a new research question requires it.
- Report performance only for kernels that also pass the irregular audit.
- Treat SASS tokens as feature evidence, not as a correctness or performance claim.
- The later exploratory runs remain useful for diagnosing failure modes, but they should not replace the P0-P3 baseline in the formal story.
