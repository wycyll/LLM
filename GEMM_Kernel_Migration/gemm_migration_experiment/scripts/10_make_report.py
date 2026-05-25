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
            f"- Performance was measured for {len(performance_cases)} kernels selected after the irregular-shape audit, producing {len(performance_rows)} shape-level rows. Treat `perf_valid` as an audit-pass selected subset, not as a measurement over every aligned-correct kernel."
        )
    if p4_cases and not p4_pass and not p4_perf:
        p4_reason = " The audit logs include `misaligned address` failures." if p4_misaligned else ""
        lines.append(
            f"- `p4_hopper_compile_safe` has no performance rows because it passed 0/{len(p4_cases)} irregular-audit cases and was filtered out before formal timing.{p4_reason} Do not interpret its `perf_valid=0` as a measured slow kernel."
        )
    return "\n".join(lines)


def workflow_q2(prompts: set[str]) -> str:
    if "p0_no_hw_hint" in prompts:
        return "Use rows with `prompt_id=p0_no_hw_hint` to compare compile/correctness rates and whether static/SASS target features appear. If P0 compiles but has no H100 WGMMA/TMA evidence, it is a runnable migration rather than a Hopper-style migration."
    return "This run does not include `p0_no_hw_hint`, so it cannot answer the no-hardware-hint question by itself. Use the P0-containing baseline runs for that comparison; this run is scoped to target-example, compile-safe, and KernelWiki-informed H100 prompts."


def workflow_q3(prompts: set[str]) -> str:
    baseline_prompts = {"p0_no_hw_hint", "p1_target_name_only", "p2_hw_feature_table", "p3_target_example"}
    if baseline_prompts.issubset(prompts):
        return "Compare P1, P2, and P3 against P0 in the table above. A useful hint should improve compile/correctness/performance or increase confirmed target-feature evidence, not only increase keyword frequency."
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

{caveats_section}

## Workflow Questions

### 1. Which hardware features matter for V100/A100/H100 GEMM?

- V100: WMMA or `mma.sync`, Volta Tensor Cores, shared-memory tiling, coalesced loads.
- A100: `cp.async`, multi-stage global-to-shared pipelines, `ldmatrix`, `mma.sync`, Ampere Tensor Cores.
- H100: TMA, WGMMA, `mbarrier`, warpgroup execution, producer-consumer warp specialization.

### 2. What happens without hardware hints?

{workflow_q2(prompts)}

### 3. Do hardware hints and examples help?

{workflow_q3(prompts)}

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
