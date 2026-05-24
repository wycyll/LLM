#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re

from common import ROOT, append_csv, command_available, filter_csv, normalize_target, read_csv, rel, run_command, write_text


FIELDNAMES = [
    "run_id",
    "task_id",
    "prompt_id",
    "sample_id",
    "target_gpu",
    "sass_has_hmma",
    "sass_has_mma",
    "sass_has_cp_async",
    "sass_has_wgmma",
    "sass_has_tma",
    "sass_has_ldmatrix",
    "sass_log_path",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan SASS/disassembly for hardware features.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--target_gpu")
    parser.add_argument("--task")
    parser.add_argument("--prompt")
    parser.add_argument("--sample")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    tool = "cuobjdump" if command_available("cuobjdump") else "nvdisasm" if command_available("nvdisasm") else ""
    if not tool:
        raise SystemExit("neither cuobjdump nor nvdisasm is available")
    target_filter = normalize_target(args.target_gpu) if args.target_gpu else None
    task_filter = filter_csv(args.task)
    prompt_filter = filter_csv(args.prompt)
    sample_filter = filter_csv(args.sample)
    rows = []
    for row in read_csv(ROOT / "results" / args.run_id / "compile_results.csv"):
        if row.get("compile_success", "").lower() != "true":
            continue
        if target_filter and normalize_target(row.get("target_gpu", "")) != target_filter:
            continue
        if task_filter and row["task_id"] not in task_filter:
            continue
        if prompt_filter and row["prompt_id"] not in prompt_filter:
            continue
        if sample_filter and row["sample_id"] not in sample_filter:
            continue
        binary = ROOT / row["binary_path"]
        log_path = ROOT / "logs" / args.run_id / "sass" / f"{row['task_id']}_{row['prompt_id']}_{row['sample_id']}.sass"
        command = [tool, "--dump-sass", str(binary)] if tool == "cuobjdump" else [tool, str(binary)]
        code, log, _elapsed = run_command(command, timeout=args.timeout)
        write_text(log_path, log)
        text = log.upper()
        rows.append(
            {
                "run_id": args.run_id,
                "task_id": row["task_id"],
                "prompt_id": row["prompt_id"],
                "sample_id": row["sample_id"],
                "target_gpu": row["target_gpu"],
                "sass_has_hmma": bool(re.search(r"\bHMMA\b", text)),
                "sass_has_mma": bool(re.search(r"\bMMA\b|MMA\.SYNC", text)),
                "sass_has_cp_async": "CP_ASYNC" in text or "CPASYNC" in text,
                "sass_has_wgmma": "WGMMA" in text,
                "sass_has_tma": "TMA" in text or "BULK" in text,
                "sass_has_ldmatrix": "LDSM" in text or "LDMATRIX" in text,
                "sass_log_path": rel(log_path),
            }
        )
    append_csv(ROOT / "results" / args.run_id / "sass_feature_results.csv", FIELDNAMES, rows)
    print(f"recorded {len(rows)} sass feature rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
