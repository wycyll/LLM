#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from common import ROOT, append_csv, case_key, ensure_run_dirs, filter_csv, normalize_target, read_csv, rel, run_command, write_text, truthy


FIELDNAMES = [
    "run_id",
    "task_id",
    "prompt_id",
    "sample_id",
    "target_gpu",
    "shape_id",
    "M",
    "N",
    "K",
    "mean_ms",
    "median_ms",
    "min_ms",
    "std_ms",
    "tflops_mean",
    "tflops_max",
    "cublas_mean_ms",
    "cublas_min_ms",
    "cublas_tflops",
    "native_ref_mean_ms",
    "native_ref_tflops",
    "speedup_vs_cublas",
    "speedup_vs_native_ref",
    "performance_valid",
    "timing_mode",
    "warmup",
    "repeat",
    "log_path",
]


def correctness_ok(rows, include_partial: bool) -> set[tuple[str, str, str, str]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[case_key(row)].append(row)
    ok = set()
    for key, values in grouped.items():
        correct_count = sum(1 for value in values if truthy(value.get("correct")))
        if include_partial and correct_count > 0:
            ok.add(key)
        elif values and correct_count == len(values):
            ok.add(key)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run performance for compiled and correct kernels.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--target_gpu", required=True)
    parser.add_argument("--shapes", default="configs/shapes_performance.json")
    parser.add_argument("--task")
    parser.add_argument("--prompt")
    parser.add_argument("--sample")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--timing_mode", choices=["formal", "smoke"], default="formal")
    parser.add_argument("--include_partial", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    ensure_run_dirs(args.run_id)
    target_gpu = normalize_target(args.target_gpu)
    compile_rows = read_csv(ROOT / "results" / args.run_id / "compile_results.csv")
    correctness_rows = read_csv(ROOT / "results" / args.run_id / "correctness_results.csv")
    ok_cases = correctness_ok(correctness_rows, args.include_partial)
    task_filter = filter_csv(args.task)
    prompt_filter = filter_csv(args.prompt)
    sample_filter = filter_csv(args.sample)
    output_rows = []
    for row in compile_rows:
        if normalize_target(row.get("target_gpu", "")) != target_gpu:
            continue
        if task_filter and row["task_id"] not in task_filter:
            continue
        if prompt_filter and row["prompt_id"] not in prompt_filter:
            continue
        if sample_filter and row["sample_id"] not in sample_filter:
            continue
        if row.get("compile_success", "").lower() != "true":
            continue
        if case_key(row) not in ok_cases:
            continue
        binary = ROOT / row["binary_path"]
        log_path = ROOT / "logs" / args.run_id / "performance" / f"{row['task_id']}_{row['prompt_id']}_{row['sample_id']}.jsonl"
        command = [
            str(binary),
            "--mode",
            "performance",
            "--shapes",
            str(ROOT / args.shapes),
            "--jsonl-output",
            str(log_path),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
        ]
        code, log, _elapsed = run_command(command, timeout=args.timeout)
        if log:
            write_text(log_path.with_suffix(".stdout.log"), log)
        if code != 0:
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            if result.get("mode") != "performance":
                continue
            output_rows.append(
                {
                    **result,
                    "run_id": args.run_id,
                    "task_id": row["task_id"],
                    "prompt_id": row["prompt_id"],
                    "sample_id": row["sample_id"],
                    "target_gpu": target_gpu,
                    "native_ref_mean_ms": "",
                    "native_ref_tflops": "",
                    "speedup_vs_native_ref": "",
                    "timing_mode": args.timing_mode,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "log_path": rel(log_path),
                }
            )
    append_csv(ROOT / "results" / args.run_id / "performance_results.csv", FIELDNAMES, output_rows)
    print(f"recorded {len(output_rows)} performance rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
