#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from common import ROOT, read_csv, truthy, write_text


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
        static_ok = sum(1 for row in values if any(token in row.get("static_features", "") for token in ["cp_async", "wgmma", "tma", "mbarrier", "warpgroup"]))
        sass_ok = sum(1 for row in attempted if any(token in row.get("sass_features", "") for token in ["cp_async", "wgmma", "tma", "ldmatrix", "hmma", "mma"]))
        lines.append(
            f"| {task} | {prompt} | {total} | {attempted_count}/{total} | {pct_or_not_run(compile_ok, attempted_count)} | {pct_or_not_run(correct_ok, attempted_count)} | {pct_or_not_run(perf_ok, attempted_count)} | {pct(static_ok, total)} | {pct_or_not_run(sass_ok, attempted_count)} |"
        )
    return "\n".join(lines)


def audit_shape_count() -> int:
    path = ROOT / "configs" / "shapes_correctness_audit.json"
    if not path.exists():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")))


def correctness_audit_section(run_id: str) -> str:
    audit_rows = read_csv(ROOT / "results" / run_id / "correctness_audit_results.csv")
    if not audit_rows:
        return ""
    expected_shapes = audit_shape_count()
    grouped_rows = grouped(audit_rows, ["task_id", "prompt_id", "sample_id"])
    pass_cases = set()
    for key, values in grouped_rows.items():
        correct_rows = [row for row in values if truthy(row.get("run_success")) and truthy(row.get("correct"))]
        if expected_shapes and len(correct_rows) == expected_shapes and len(values) == expected_shapes:
            pass_cases.add(key)
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
        if any(token in row.get("static_features", "") for token in ["wgmma", "tma", "cp_async"])
        and row.get("compile_status") != "not_run"
        and not any(token in row.get("sass_features", "") for token in ["wgmma", "tma", "cp_async"])
    )
    audit_section = correctness_audit_section(args.run_id)

    report = f"""# GEMM Migration Report: {args.run_id}

## Executive Summary

- Samples generated: {total}
- Target-GPU compile attempted: {attempted_count}/{total} ({pct(attempted_count, total)})
- Compile success among attempted: {compile_ok}/{attempted_count} ({pct_or_not_run(compile_ok, attempted_count)})
- Aligned correctness shapes passed among attempted: {correct_ok}/{attempted_count} ({pct_or_not_run(correct_ok, attempted_count)})
- Performance measured among attempted: {perf_ok}/{attempted_count} ({pct_or_not_run(perf_ok, attempted_count)})
- Static target-feature claim without SASS confirmation among attempted samples: {claimed_not_confirmed}/{attempted_count} ({pct_or_not_run(claimed_not_confirmed, attempted_count)})

## Results By Task And Prompt

{table_by_task_prompt(rows)}

{audit_section}

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
"""

    write_text(ROOT / "results" / args.run_id / "summary.md", report)
    write_text(ROOT / "reports" / args.run_id / "final_report.md", report)
    print(f"wrote results/{args.run_id}/summary.md and reports/{args.run_id}/final_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
