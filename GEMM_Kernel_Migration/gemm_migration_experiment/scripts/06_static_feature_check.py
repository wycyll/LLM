#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re

from common import ROOT, append_csv, filter_csv, iter_cases, rel, read_text


FIELDNAMES = [
    "run_id",
    "task_id",
    "prompt_id",
    "sample_id",
    "source_path",
    "has_shared_memory",
    "has_wmma",
    "has_mma_sync",
    "has_cp_async",
    "has_ldmatrix",
    "has_tma",
    "has_wgmma",
    "has_mbarrier",
    "has_warpgroup",
    "has_producer_consumer",
]


PATTERNS = {
    "has_shared_memory": r"__shared__|extern\s+__shared__",
    "has_wmma": r"nvcuda::wmma|wmma::",
    "has_mma_sync": r"mma\.sync|wmma::mma_sync",
    "has_cp_async": r"cp\.async|cuda::memcpy_async",
    "has_ldmatrix": r"ldmatrix|LDMATRIX",
    "has_tma": r"cp\.async\.bulk\.tensor|tma|TMA|CUtensorMap|cuda::memcpy_async",
    "has_wgmma": r"wgmma|WGMMA|wgmma\.mma_async",
    "has_mbarrier": r"mbarrier|barrier\.arrive|barrier\.wait",
    "has_warpgroup": r"warpgroup|warp-group|warp group",
    "has_producer_consumer": r"producer|consumer|load_warp|compute_warp|role",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Static source feature scan.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--tasks")
    parser.add_argument("--prompts")
    parser.add_argument("--samples")
    args = parser.parse_args()

    rows = []
    for case in iter_cases(
        args.run_id,
        task_filter=filter_csv(args.tasks),
        prompt_filter=filter_csv(args.prompts),
        sample_filter=filter_csv(args.samples),
    ):
        source = case.sample_dir / "kernel_extracted.cu"
        if not source.exists():
            continue
        text = read_text(source)
        row = {
            "run_id": args.run_id,
            "task_id": case.task_id,
            "prompt_id": case.prompt_id,
            "sample_id": case.sample_id,
            "source_path": rel(source),
        }
        for field, pattern in PATTERNS.items():
            row[field] = bool(re.search(pattern, text, flags=re.IGNORECASE))
        rows.append(row)
    append_csv(ROOT / "results" / args.run_id / "static_feature_results.csv", FIELDNAMES, rows)
    print(f"recorded {len(rows)} static feature rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
