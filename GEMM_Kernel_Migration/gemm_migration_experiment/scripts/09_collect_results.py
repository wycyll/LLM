#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict

from common import ROOT, append_csv, case_key, iter_cases, read_csv, truthy


FIELDNAMES = [
    "run_id",
    "task_id",
    "prompt_id",
    "sample_id",
    "source_gpu",
    "target_gpu",
    "target_arch",
    "compile_status",
    "compile_success",
    "correctness_status",
    "performance_status",
    "mean_tflops",
    "speedup_vs_cublas",
    "speedup_vs_native_ref",
    "static_features",
    "sass_features",
    "profile_features",
    "migration_quality_type",
    "main_failure_reason",
    "notes",
]


def list_features(row: dict, prefix: str) -> str:
    features = []
    for key, value in row.items():
        if key.startswith(prefix) and truthy(value):
            features.append(key.removeprefix(prefix))
    return ";".join(sorted(features))


def correctness_status(rows: list[dict]) -> str:
    if not rows:
        return "not_run"
    run_ok = [truthy(row.get("run_success")) for row in rows]
    correct = [truthy(row.get("correct")) for row in rows]
    if not all(run_ok):
        return "runtime_failed"
    if all(correct):
        return "all_correct"
    if any(correct):
        return "partial_correct"
    return "wrong_result"


def target_feature_present(target_gpu: str, static: str, sass: str) -> bool:
    target = target_gpu.lower()
    combined = static + ";" + sass
    if "h100" in target:
        return any(feature in combined for feature in ["wgmma", "tma", "mbarrier", "warpgroup"])
    if "a100" in target:
        return "cp_async" in combined or "ldmatrix" in combined or "mma_sync" in combined
    if "v100" in target:
        return "wmma" in combined or "hmma" in combined or "mma" in combined
    return False


def classify_quality(compile_status: str, compile_success: bool, correctness: str, perf_rows: list[dict], static: str, sass: str, target_gpu: str) -> tuple[str, str]:
    if compile_status == "not_run":
        return "Not evaluated", "target_gpu_not_run"
    if not compile_success:
        return "Type 0", "compile_error"
    if correctness == "runtime_failed":
        return "Type 1", "runtime_crash"
    if correctness in {"wrong_result", "partial_correct"}:
        return "Type 2", "wrong_result" if correctness == "wrong_result" else "partial_shape_support"
    if correctness == "not_run":
        return "Type 1", "not_run"
    if not perf_rows:
        return "Type 4", "performance_not_run"
    has_target = target_feature_present(target_gpu, static, sass)
    best_tflops = max(float(row.get("tflops_mean") or 0.0) for row in perf_rows)
    if best_tflops <= 0.0:
        return "Type 3", "too_slow"
    if has_target:
        return "Type 5", "needs_native_ref_ratio"
    if "shared_memory" in static:
        return "Type 4", "no_target_specific_feature"
    return "Type 3", "no_target_specific_feature"


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge raw CSVs into migration_quality.csv.")
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    compile_by_key = {case_key(row): row for row in read_csv(ROOT / "results" / args.run_id / "compile_results.csv")}
    corr_by_key = defaultdict(list)
    for row in read_csv(ROOT / "results" / args.run_id / "correctness_results.csv"):
        corr_by_key[case_key(row)].append(row)
    perf_by_key = defaultdict(list)
    for row in read_csv(ROOT / "results" / args.run_id / "performance_results.csv"):
        perf_by_key[case_key(row)].append(row)
    static_by_key = {case_key(row): row for row in read_csv(ROOT / "results" / args.run_id / "static_feature_results.csv")}
    sass_by_key = {case_key(row): row for row in read_csv(ROOT / "results" / args.run_id / "sass_feature_results.csv")}
    profile_by_key = {case_key(row): row for row in read_csv(ROOT / "results" / args.run_id / "profile_results.csv")}

    keys = set(compile_by_key) | set(corr_by_key) | set(perf_by_key) | set(static_by_key) | set(sass_by_key)
    for case in iter_cases(args.run_id):
        keys.add((case.run_id, case.task_id, case.prompt_id, case.sample_id))

    rows = []
    for key in sorted(keys):
        run_id, task_id, prompt_id, sample_id = key
        compile_row = compile_by_key.get(key, {})
        compile_status = "not_run"
        if key in compile_by_key:
            compile_status = "success" if truthy(compile_row.get("compile_success")) else "failed"
        case_metadata = {}
        metadata_path = ROOT / "generated" / run_id / task_id / prompt_id / sample_id / "metadata.json"
        if metadata_path.exists():
            import json

            case_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        compile_success = truthy(compile_row.get("compile_success")) if compile_status != "not_run" else ""
        corr_status = "not_evaluated" if compile_status == "not_run" else correctness_status(corr_by_key.get(key, []))
        perf_rows = perf_by_key.get(key, [])
        static = list_features(static_by_key.get(key, {}), "has_")
        sass = list_features(sass_by_key.get(key, {}), "sass_has_")
        profile = "ncu" if truthy(profile_by_key.get(key, {}).get("ncu_success")) else ""
        quality, failure = classify_quality(
            compile_status,
            compile_success,
            corr_status,
            perf_rows,
            static,
            sass,
            str(case_metadata.get("target_gpu") or compile_row.get("target_gpu", "")),
        )
        best_perf = max(perf_rows, key=lambda row: float(row.get("tflops_mean") or 0.0), default={})
        rows.append(
            {
                "run_id": run_id,
                "task_id": task_id,
                "prompt_id": prompt_id,
                "sample_id": sample_id,
                "source_gpu": case_metadata.get("source_gpu", ""),
                "target_gpu": case_metadata.get("target_gpu") or compile_row.get("target_gpu", ""),
                "target_arch": case_metadata.get("target_arch") or compile_row.get("target_arch", ""),
                "compile_status": compile_status,
                "compile_success": compile_success,
                "correctness_status": corr_status,
                "performance_status": "not_evaluated" if compile_status == "not_run" else "valid" if perf_rows else "not_run",
                "mean_tflops": best_perf.get("tflops_mean", ""),
                "speedup_vs_cublas": best_perf.get("speedup_vs_cublas", ""),
                "speedup_vs_native_ref": best_perf.get("speedup_vs_native_ref", ""),
                "static_features": static,
                "sass_features": sass,
                "profile_features": profile,
                "migration_quality_type": quality,
                "main_failure_reason": failure,
                "notes": "Type 6/7 require native_ref_tflops to be filled from expert references.",
            }
        )
    output = ROOT / "results" / args.run_id / "migration_quality.csv"
    if output.exists():
        output.unlink()
    append_csv(output, FIELDNAMES, rows)
    print(f"wrote {len(rows)} migration quality rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
