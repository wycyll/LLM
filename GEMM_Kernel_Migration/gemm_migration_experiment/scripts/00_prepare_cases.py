#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from common import ROOT, ensure_run_dirs, filter_csv, load_matrix, read_text, write_json, write_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Create generated/<run_id> case directories.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--tasks", help="Comma-separated task IDs. Defaults to required tasks.")
    parser.add_argument("--prompts", help="Comma-separated prompt IDs. Defaults to matrix prompts.")
    parser.add_argument("--samples", type=int, help="Samples per prompt. Defaults to matrix setting.")
    parser.add_argument("--include_optional", action="store_true", help="Prepare optional tasks that have complete metadata.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    matrix = load_matrix()
    task_filter = filter_csv(args.tasks)
    prompt_filter = filter_csv(args.prompts)
    all_tasks = list(matrix.get("required_tasks", []))
    optional_tasks = list(matrix.get("optional_tasks", []))
    if task_filter or args.include_optional:
        all_tasks.extend(optional_tasks)
    tasks = [task for task in all_tasks if not task_filter or task["task_id"] in task_filter]
    tasks = [task for task in tasks if task.get("source_kernel_path") and task.get("target_arch")]
    prompts = [prompt for prompt in matrix.get("prompts", []) if not prompt_filter or prompt in prompt_filter]
    samples = args.samples or int(matrix.get("samples_per_prompt", 3))
    generated_at = datetime.now(timezone.utc).isoformat()

    ensure_run_dirs(args.run_id)
    count = 0
    for task in tasks:
        for prompt in prompts:
            for sample_index in range(samples):
                sample_id = f"sample_{sample_index:02d}"
                sample_dir = ROOT / "generated" / args.run_id / task["task_id"] / prompt / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = sample_dir / "metadata.json"
                if metadata_path.exists() and not args.overwrite:
                    continue
                metadata = {
                    "run_id": args.run_id,
                    "task_id": task["task_id"],
                    "prompt_id": prompt,
                    "sample_id": sample_id,
                    "model": "",
                    "temperature": "",
                    "source_gpu": task.get("source_gpu", ""),
                    "target_gpu": task.get("target_gpu", ""),
                    "source_key": task.get("source_key", ""),
                    "target_key": task.get("target_key", ""),
                    "source_arch": task.get("source_arch", ""),
                    "target_arch": task.get("target_arch", ""),
                    "source_kernel_path": task.get("source_kernel_path", ""),
                    "source_launch_path": task.get("source_launch_path", ""),
                    "generated_at": generated_at,
                    "manual_core_patch": False,
                    "wrapper_only": False,
                    "response_status": "pending",
                }
                write_json(metadata_path, metadata)
                if args.overwrite or not (sample_dir / "response_raw.md").exists():
                    write_text(sample_dir / "response_raw.md", "")
                count += 1

    print(f"prepared {count} cases under generated/{args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
