#!/usr/bin/env python3

from __future__ import annotations

import argparse

from common import ROOT, append_csv, case_key, command_available, read_csv, rel, run_command, write_text


FIELDNAMES = [
    "run_id",
    "task_id",
    "prompt_id",
    "sample_id",
    "target_gpu",
    "selected_reason",
    "ncu_success",
    "ncu_report_path",
    "ncu_log_path",
]


def select_cases(run_id: str, selected: str, limit: int):
    compile_rows = [row for row in read_csv(ROOT / "results" / run_id / "compile_results.csv") if row.get("compile_success", "").lower() == "true"]
    if selected == "all_compiled":
        return compile_rows[:limit]
    perf_rows = read_csv(ROOT / "results" / run_id / "performance_results.csv")
    best = {}
    for row in perf_rows:
        key = case_key(row)
        score = float(row.get("tflops_mean") or 0.0)
        if key not in best or score > best[key][0]:
            best[key] = (score, row)
    sorted_keys = [key for key, _value in sorted(best.items(), key=lambda item: item[1][0], reverse=True)]
    selected_compile = []
    compile_by_key = {case_key(row): row for row in compile_rows}
    for key in sorted_keys:
        if key in compile_by_key:
            selected_compile.append(compile_by_key[key])
        if len(selected_compile) >= limit:
            break
    return selected_compile


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Nsight Compute for selected kernels.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--selected", choices=["top_correct_kernels", "all_compiled"], default="top_correct_kernels")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--shapes", default="configs/shapes_performance.json")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not command_available("ncu") and not args.dry_run:
        raise SystemExit("ncu is not available")
    rows = []
    for row in select_cases(args.run_id, args.selected, args.limit):
        binary = ROOT / row["binary_path"]
        report_path = ROOT / "logs" / args.run_id / "ncu" / f"{row['task_id']}_{row['prompt_id']}_{row['sample_id']}.ncu-rep"
        log_path = report_path.with_suffix(".log")
        command = [
            "ncu",
            "--set",
            "full",
            "--target-processes",
            "all",
            "--force-overwrite",
            "--export",
            str(report_path),
            str(binary),
            "--mode",
            "performance",
            "--shapes",
            str(ROOT / args.shapes),
            "--warmup",
            "1",
            "--repeat",
            "3",
        ]
        if args.dry_run:
            write_text(log_path, "$ " + " ".join(command) + "\n")
            code = 0
            log = "dry run"
        else:
            code, log, _elapsed = run_command(command, timeout=args.timeout)
            write_text(log_path, log)
        rows.append(
            {
                "run_id": args.run_id,
                "task_id": row["task_id"],
                "prompt_id": row["prompt_id"],
                "sample_id": row["sample_id"],
                "target_gpu": row["target_gpu"],
                "selected_reason": args.selected,
                "ncu_success": code == 0,
                "ncu_report_path": rel(report_path),
                "ncu_log_path": rel(log_path),
            }
        )
    append_csv(ROOT / "results" / args.run_id / "profile_results.csv", FIELDNAMES, rows)
    print(f"recorded {len(rows)} profile rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
