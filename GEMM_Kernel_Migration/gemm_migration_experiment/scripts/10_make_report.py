#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from common import ROOT, read_csv, truthy, write_text


TARGET_STATIC_TOKENS = ["cp_async", "wgmma", "tma", "mbarrier", "warpgroup"]
TARGET_SASS_TOKENS = ["cp_async", "wgmma", "tma", "ldmatrix", "hmma", "mma"]
SUMMARY_CONFIRM_TOKENS = ["wgmma", "tma", "cp_async"]


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def pct_or_not_run(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "not run"
    return pct(numerator, denominator)


def grouped(rows, key_fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row.get(field, "") for field in key_fields)].append(row)
    return result


def table_by_task_prompt(rows):
    lines = [
        "| task | prompt | samples | compile_attempted | compile | aligned_all_correct | perf_valid | target_static | target_sass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (task, prompt), values in sorted(grouped(rows, ["task_id", "prompt_id"]).items()):
        total = len(values)
        attempted = [row for row in values if row.get("compile_status") != "not_run"]
        attempted_count = len(attempted)
        compile_ok = sum(1 for row in attempted if truthy(row.get("compile_success")))
        correct_ok = sum(1 for row in attempted if row.get("correctness_status") == "all_correct")
        perf_ok = sum(1 for row in attempted if row.get("performance_status") == "valid")
        static_ok = sum(1 for row in values if any(token in row.get("static_features", "") for token in TARGET_STATIC_TOKENS))
        sass_ok = sum(1 for row in attempted if any(token in row.get("sass_features", "") for token in TARGET_SASS_TOKENS))
        lines.append(
            f"| {task} | {prompt} | {total} | {attempted_count}/{total} | {pct_or_not_run(compile_ok, attempted_count)} | {pct_or_not_run(correct_ok, attempted_count)} | {pct_or_not_run(perf_ok, attempted_count)} | {pct(static_ok, total)} | {pct_or_not_run(sass_ok, attempted_count)} |"
        )
    return "\n".join(lines)


def audit_pass_cases(run_id: str):
    audit_rows = read_csv(ROOT / "results" / run_id / "correctness_audit_results.csv")
    expected_shapes = audit_shape_count()
    grouped_rows = grouped(audit_rows, ["task_id", "prompt_id", "sample_id"])
    pass_cases = set()
    for key, values in grouped_rows.items():
        correct_rows = [row for row in values if truthy(row.get("run_success")) and truthy(row.get("correct"))]
        if expected_shapes and len(correct_rows) == expected_shapes and len(values) == expected_shapes:
            pass_cases.add(key)
    return audit_rows, grouped_rows, pass_cases


def audit_shape_count() -> int:
    path = ROOT / "configs" / "shapes_correctness_audit.json"
    if not path.exists():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")))


def correctness_audit_section(run_id: str) -> str:
    audit_rows, grouped_rows, pass_cases = audit_pass_cases(run_id)
    if not audit_rows:
        return ""
    lines = [
        "## Irregular-Shape Correctness Audit",
        "",
        "This section uses `configs/shapes_correctness_audit.json` to stress non-tile-multiple boundary cases. It is stricter than the aligned correctness suite above.",
        "",
        "| task | prompt | compiled_audited | audit_all_correct |",
        "| --- | --- | ---: | ---: |",
    ]
    by_task_prompt = defaultdict(list)
    for task_id, prompt_id, sample_id in grouped_rows:
        by_task_prompt[(task_id, prompt_id)].append((task_id, prompt_id, sample_id))
    total_cases = len(grouped_rows)
    total_pass = len(pass_cases)
    lines.insert(3, f"Overall audit pass: {total_pass}/{total_cases} ({pct(total_pass, total_cases)}).")
    lines.insert(4, "")
    for (task_id, prompt_id), cases in sorted(by_task_prompt.items()):
        passed = sum(1 for case in cases if case in pass_cases)
        lines.append(f"| {task_id} | {prompt_id} | {len(cases)} | {passed}/{len(cases)} ({pct(passed, len(cases))}) |")
    return "\n".join(lines)


def run_scope_and_caveats_section(run_id: str, rows, claimed_not_confirmed: int, attempted_count: int) -> str:
    prompts = sorted({row.get("prompt_id", "") for row in rows if row.get("prompt_id")})
    missing_baseline_prompts = [
        prompt
        for prompt in ["p0_no_hw_hint", "p1_target_name_only", "p2_hw_feature_table"]
        if prompt not in prompts
    ]
    performance_rows = read_csv(ROOT / "results" / run_id / "performance_results.csv")
    performance_cases = {
        (row.get("task_id", ""), row.get("prompt_id", ""), row.get("sample_id", ""))
        for row in performance_rows
    }
    audit_rows, audit_cases, pass_cases = audit_pass_cases(run_id)
    p4_cases = [case for case in audit_cases if case[1] == "p4_hopper_compile_safe"]
    p4_pass = [case for case in p4_cases if case in pass_cases]
    p4_perf = [case for case in performance_cases if case[1] == "p4_hopper_compile_safe"]
    p4_misaligned = any(
        row.get("prompt_id") == "p4_hopper_compile_safe" and "misaligned address" in row.get("runtime_error", "")
        for row in audit_rows
    )

    lines = [
        "## Run Scope And Caveats",
        "",
        f"- This run covers these prompt IDs only: `{', '.join(prompts)}`.",
    ]
    if missing_baseline_prompts:
        lines.append(
            "- It does not include the baseline prompts "
            + ", ".join(f"`{prompt}`" for prompt in missing_baseline_prompts)
            + ". Use P0/P1/P2-containing runs for no-hardware-hint, target-name-only, and hardware-feature-table conclusions."
        )
    lines.extend(
        [
            f"- The summary count `{claimed_not_confirmed}/{attempted_count}` is a row-level rule: a sample is counted when `static_features` contains one of `{', '.join(SUMMARY_CONFIRM_TOKENS)}` but `sass_features` contains none of those same tokens. It is not computed by subtracting aggregate table percentages, and one sample can contain multiple static features.",
            "- `target_sass` is evidence that selected instruction-family tokens appear in SASS. It is not a performance claim and does not imply the kernel is correct, robust, or fast.",
            "- Nsight Compute profile evidence is unavailable for this run, so hardware-feature confirmation here is limited to static source scanning and SASS inspection.",
        ]
    )
    if performance_rows and audit_rows:
        lines.append(
            f"- Performance rows exist for {len(performance_cases)} kernels, producing {len(performance_rows)} shape-level rows. Interpret TFLOPS only after cross-checking the irregular-shape audit; aligned-correct but audit-failing kernels should not be reported as generally correct."
        )
    if p4_cases and not p4_pass and not p4_perf:
        p4_reason = " The audit logs include `misaligned address` failures." if p4_misaligned else ""
        lines.append(
            f"- `p4_hopper_compile_safe` has no performance rows because it passed 0/{len(p4_cases)} irregular-audit cases and was filtered out before formal timing.{p4_reason} Do not interpret its `perf_valid=0` as a measured slow kernel."
        )
    return "\n".join(lines)


def baseline_ablation_section(run_id: str, rows) -> str:
    prompt_order = ["p0_no_hw_hint", "p1_target_name_only", "p2_hw_feature_table", "p3_target_example"]
    prompts = {row.get("prompt_id", "") for row in rows}
    if not set(prompt_order).issubset(prompts):
        return ""

    _audit_rows, audit_cases, pass_cases = audit_pass_cases(run_id)
    labels = {
        "p0_no_hw_hint": "no hardware hint",
        "p1_target_name_only": "target name only",
        "p2_hw_feature_table": "hardware feature table",
        "p3_target_example": "target example",
    }
    lines = [
        "## Baseline Prompt Ablation Findings",
        "",
        "This section answers the baseline prompt question using only P0-P3. The stricter decision signal is the irregular-shape audit, not aligned-shape correctness alone.",
        "",
        "| prompt | condition | compile | aligned_all_correct | irregular_audit | best_audit_pass_TFLOPS | interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for prompt in prompt_order:
        values = [row for row in rows if row.get("prompt_id") == prompt]
        attempted = [row for row in values if row.get("compile_status") != "not_run"]
        compiled = [row for row in attempted if truthy(row.get("compile_success"))]
        aligned_ok = [row for row in attempted if row.get("correctness_status") == "all_correct"]
        prompt_audit_cases = [case for case in audit_cases if case[1] == prompt]
        prompt_pass_cases = [case for case in prompt_audit_cases if case in pass_cases]
        best_audit = max(
            [
                float(row.get("mean_tflops", "0"))
                for row in values
                if row.get("mean_tflops") and (row.get("task_id", ""), row.get("prompt_id", ""), row.get("sample_id", "")) in pass_cases
            ]
            or [0.0]
        )
        best_text = f"{best_audit:.3f}" if best_audit else "-"
        if prompt == "p0_no_hw_hint":
            interpretation = (
                "Some robust candidates exist, but success is target- and sample-dependent; no-hint migration is not reliable by itself."
                if prompt_pass_cases
                else "No reliable migration signal: target-looking code may appear, but this run has no irregular-audit-pass P0 candidate."
            )
        elif prompt == "p1_target_name_only":
            interpretation = (
                "Target naming can help some samples, but aligned-suite success or TFLOPS must still be discounted when irregular audit fails."
                if prompt_pass_cases
                else "Target name alone is weak: aligned-suite success or TFLOPS must be discounted if irregular audit fails."
            )
        elif prompt == "p2_hw_feature_table":
            interpretation = (
                "The feature table can produce an occasional robust sample, but it also increases feature claims and compile instability."
                if prompt_pass_cases
                else "The feature table does not help by itself here; it increases feature claims but hurts compile stability."
            )
        else:
            interpretation = "Target examples are the most reliable baseline prompt overall, but passing examples still do not prove WGMMA/TMA-style target optimization."
        lines.append(
            f"| {prompt} | {labels[prompt]} | {len(compiled)}/{len(attempted)} | {len(aligned_ok)}/{len(compiled) if compiled else len(attempted)} | {len(prompt_pass_cases)}/{len(prompt_audit_cases)} | {best_text} | {interpretation} |"
        )
    evaluated_tasks = sorted({row.get("task_id", "") for row in rows if row.get("compile_status") != "not_run"})
    evaluated_targets = sorted({task.split("_to_")[-1] for task in evaluated_tasks if "_to_" in task})
    if evaluated_targets == ["h100"]:
        answer = "Answer: in this H100 baseline, hardware hints help only when they include a target-style example. Hardware name alone and feature-table text mainly produce claims or aligned-suite false positives; they do not produce robust H100 GEMM migration evidence."
    else:
        answer = "Answer: in this multi-target baseline, hardware hints are not a single monotonic improvement. No-hint and target-name prompts produce some audit-pass A100 samples, the feature table has only isolated success with compile instability, and target examples are strongest overall but not uniformly robust across target directions. Report the effect per task/prompt, and treat target examples as useful guidance rather than proof of hardware-optimal migration."
    lines.extend(["", answer])
    return "\n".join(lines)


def hardware_fit_section() -> str:
    return """## Hardware Fit And Migration Boundary

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
| Performance-portable architecture selection | Not demonstrated. The model does not reliably choose the right hardware-specific strategy or tune tile sizes/stages/occupancy from hardware names or feature tables alone. |"""


def workflow_q2(run_id: str, prompts: set[str], rows) -> str:
    if "p0_no_hw_hint" in prompts:
        evaluated_tasks = sorted({row.get("task_id", "") for row in rows if row.get("compile_status") != "not_run"})
        evaluated_targets = sorted({task.split("_to_")[-1] for task in evaluated_tasks if "_to_" in task})
        if evaluated_targets == ["h100"]:
            return "Use the P0 row in the baseline ablation table above. In this P0-P3 H100 baseline, no-hardware-hint generation does not produce a robust irregular-audit-pass H100 migration."
        return "Use the P0 row in the baseline ablation table above. In this multi-target baseline, no-hardware-hint generation can produce some audit-pass candidates, but success is task- and target-dependent and should not be treated as reliable hardware-style migration."
    return "This run does not include `p0_no_hw_hint`, so it cannot answer the no-hardware-hint question by itself. Use the P0-containing baseline runs for that comparison; this run is scoped to target-example, compile-safe, and KernelWiki-informed H100 prompts."


def workflow_q3(run_id: str, prompts: set[str], rows) -> str:
    baseline_prompts = {"p0_no_hw_hint", "p1_target_name_only", "p2_hw_feature_table", "p3_target_example"}
    if baseline_prompts.issubset(prompts):
        evaluated_tasks = sorted({row.get("task_id", "") for row in rows if row.get("compile_status") != "not_run"})
        evaluated_targets = sorted({task.split("_to_")[-1] for task in evaluated_tasks if "_to_" in task})
        if evaluated_targets != ["h100"]:
            return "In this multi-target baseline, hints and examples help unevenly. Target examples give the strongest overall robustness signal, especially for H100, but they are not uniformly correct across A100 back-migration tasks. Target-name and feature-table prompts can create some audit-pass samples, yet they also produce false positives and compile instability."
        return "In this baseline, the target example P3 is the only prompt that reliably improves robustness. The target-name-only P1 and hardware-feature-table P2 variants mainly produce feature claims, compile failures, or aligned-suite false positives, not verified Hopper-style GEMM."
    return "This run compares the available prompt variants in the table above. For the original P0/P1/P2/P3 hint ablation, use a run that contains all four baseline prompts; run04 mainly tests whether compile-safe and KernelWiki-informed prompts improve robustness or feature evidence relative to P3/P4-style prompts."


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Markdown summary report.")
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    rows = read_csv(ROOT / "results" / args.run_id / "migration_quality.csv")
    if not rows:
        raise SystemExit("migration_quality.csv is missing or empty; run 09_collect_results.py first")

    total = len(rows)
    attempted = [row for row in rows if row.get("compile_status") != "not_run"]
    attempted_count = len(attempted)
    compile_ok = sum(1 for row in attempted if truthy(row.get("compile_success")))
    correct_ok = sum(1 for row in attempted if row.get("correctness_status") == "all_correct")
    perf_ok = sum(1 for row in attempted if row.get("performance_status") == "valid")
    claimed_not_confirmed = sum(
        1
        for row in rows
        if any(token in row.get("static_features", "") for token in SUMMARY_CONFIRM_TOKENS)
        and row.get("compile_status") != "not_run"
        and not any(token in row.get("sass_features", "") for token in SUMMARY_CONFIRM_TOKENS)
    )
    audit_section = correctness_audit_section(args.run_id)
    baseline_section = baseline_ablation_section(args.run_id, rows)
    caveats_section = run_scope_and_caveats_section(args.run_id, rows, claimed_not_confirmed, attempted_count)
    prompts = {row.get("prompt_id", "") for row in rows}

    report = f"""# GEMM Migration Report: {args.run_id}

## Executive Summary

- Samples generated: {total}
- Target-GPU compile attempted: {attempted_count}/{total} ({pct(attempted_count, total)})
- Compile success among attempted: {compile_ok}/{attempted_count} ({pct_or_not_run(compile_ok, attempted_count)})
- Aligned correctness shapes passed among attempted: {correct_ok}/{attempted_count} ({pct_or_not_run(correct_ok, attempted_count)})
- Performance measured among attempted: {perf_ok}/{attempted_count} ({pct_or_not_run(perf_ok, attempted_count)})
- Static WGMMA/TMA/cp.async claim without matching SASS confirmation among attempted samples: {claimed_not_confirmed}/{attempted_count} ({pct_or_not_run(claimed_not_confirmed, attempted_count)})

## Results By Task And Prompt

{table_by_task_prompt(rows)}

{audit_section}

{baseline_section}

{hardware_fit_section()}

{caveats_section}

## Workflow Questions

### 1. Which hardware features matter for V100/A100/H100 GEMM?

- V100: WMMA or `mma.sync`, Volta Tensor Cores, shared-memory tiling, coalesced loads.
- A100: `cp.async`, multi-stage global-to-shared pipelines, `ldmatrix`, `mma.sync`, Ampere Tensor Cores.
- H100: TMA, WGMMA, `mbarrier`, warpgroup execution, producer-consumer warp specialization.

### 2. What happens without hardware hints?

{workflow_q2(args.run_id, prompts, rows)}

### 3. Do hardware hints and examples help?

{workflow_q3(args.run_id, prompts, rows)}

### 4. Do generated kernels compile, run, compute correctly, and perform?

The `compile_status`, `compile_success`, `correctness_status`, `performance_status`, `mean_tflops`, and `speedup_vs_cublas` columns in `migration_quality.csv` are the aligned-suite evidence. Use `correctness_audit_results.csv` when judging arbitrary-shape boundary correctness. Formal conclusions require target-GPU runs; samples with `compile_status=not_run` are generated but not yet evaluated on their target GPU.

### 5. Which features were only written but not verified?

Rows where `static_features` contains a target feature but `sass_features` and `profile_features` do not confirm it should be treated as `claimed_but_not_verified`. These cases are especially important for WGMMA/TMA/mbarrier claims.

## Notes

Type 6 and Type 7 require expert native-reference TFLOPS. Fill `native_ref_tflops` from validated V100/A100/H100 reference kernels before using those labels as final claims.
"""

    write_text(ROOT / "results" / args.run_id / "summary.md", report)
    write_text(ROOT / "reports" / args.run_id / "final_report.md", report)
    print(f"wrote results/{args.run_id}/summary.md and reports/{args.run_id}/final_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
