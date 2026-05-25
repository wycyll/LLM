#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from json import JSONDecodeError

from common import ROOT, append_csv, ensure_run_dirs, filter_csv, normalize_target, read_csv, rel, run_command, write_text


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
    "compiled",
    "run_success",
    "correct",
    "max_abs_error",
    "max_rel_error",
    "mean_abs_error",
    "nan_count",
    "inf_count",
    "runtime_error",
    "log_path",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stricter irregular-shape correctness audit for compiled kernels.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--target_gpu", required=True)
    parser.add_argument("--shapes", default="configs/shapes_correctness_audit.json")
    parser.add_argument("--task")
    parser.add_argument("--prompt")
    parser.add_argument("--sample")
    parser.add_argument("--rtol", default="0.01")
    parser.add_argument("--atol", default="0.01")
    parser.add_argument("--seed", default="4321")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    ensure_run_dirs(args.run_id)
    target_gpu = normalize_target(args.target_gpu)
    compile_rows = read_csv(ROOT / "results" / args.run_id / "compile_results.csv")
    task_filter = filter_csv(args.task)
    prompt_filter = filter_csv(args.prompt)
    sample_filter = filter_csv(args.sample)
    def selected_result_row(row: dict) -> bool:
        if normalize_target(row.get("target_gpu", "")) != target_gpu:
            return False
        if task_filter and row.get("task_id", "") not in task_filter:
            return False
        if prompt_filter and row.get("prompt_id", "") not in prompt_filter:
            return False
        if sample_filter and row.get("sample_id", "") not in sample_filter:
            return False
        return True

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
        binary = ROOT / row["binary_path"]
        log_path = ROOT / "logs" / args.run_id / "correctness_audit" / f"{row['task_id']}_{row['prompt_id']}_{row['sample_id']}.jsonl"
        command = [
            str(binary),
            "--mode",
            "correctness",
            "--shapes",
            str(ROOT / args.shapes),
            "--jsonl-output",
            str(log_path),
            "--rtol",
            str(args.rtol),
            "--atol",
            str(args.atol),
            "--warmup",
            "1",
            "--repeat",
            "1",
            "--seed",
            str(args.seed),
        ]
        code, log, _elapsed = run_command(command, timeout=args.timeout)
        if log:
            write_text(log_path.with_suffix(".stdout.log"), log)
        if code != 0:
            output_rows.append(
                {
                    "run_id": args.run_id,
                    "task_id": row["task_id"],
                    "prompt_id": row["prompt_id"],
                    "sample_id": row["sample_id"],
                    "target_gpu": target_gpu,
                    "compiled": True,
                    "run_success": False,
                    "correct": False,
                    "runtime_error": (log or f"runner exited {code}")[:500],
                    "log_path": rel(log_path),
                }
            )
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            json_text = line.strip()
            if not json_text.startswith("{"):
                start = json_text.find("{")
                end = json_text.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    continue
                json_text = json_text[start : end + 1]
            try:
                result = json.loads(json_text)
            except JSONDecodeError:
                continue
            if result.get("mode") != "correctness":
                continue
            output_rows.append(
                {
                    **result,
                    "run_id": args.run_id,
                    "task_id": row["task_id"],
                    "prompt_id": row["prompt_id"],
                    "sample_id": row["sample_id"],
                    "target_gpu": target_gpu,
                    "compiled": True,
                    "log_path": rel(log_path),
                }
            )
    output_path = ROOT / "results" / args.run_id / "correctness_audit_results.csv"
    preserved_rows = [row for row in read_csv(output_path) if not selected_result_row(row)] if output_path.exists() else []
    if output_path.exists():
        output_path.unlink()
    append_csv(output_path, FIELDNAMES, preserved_rows + output_rows)
    print(f"recorded {len(output_rows)} audit correctness rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())