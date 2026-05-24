#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil

from common import (
    ROOT,
    append_csv,
    classify_compile_error,
    cmake_arch,
    command_available,
    ensure_run_dirs,
    filter_csv,
    iter_cases,
    normalize_target,
    rel,
    run_command,
    summarize_error,
    update_json,
    write_text,
)


FIELDNAMES = [
    "run_id",
    "task_id",
    "prompt_id",
    "sample_id",
    "target_gpu",
    "target_arch",
    "compile_success",
    "compile_time_sec",
    "binary_path",
    "compile_log_path",
    "error_type",
    "error_summary",
]


def build_one(args, task_id: str, prompt_id: str, sample_id: str, target_gpu: str, kernel_source) -> dict:
    build_dir = ROOT / "build" / args.run_id / target_gpu / task_id / prompt_id / sample_id
    build_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / args.run_id / "compile" / f"{task_id}_{prompt_id}_{sample_id}.log"
    binary_path = build_dir / "gemm_runner"
    arch = cmake_arch(args.arch)
    generator_args = ["-G", "Ninja"] if command_available("ninja") else []
    configure = [
        "cmake",
        "-S",
        str(ROOT / "harness"),
        "-B",
        str(build_dir),
        *generator_args,
        f"-DKERNEL_SOURCE={kernel_source.resolve()}",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
    ]
    build = ["cmake", "--build", str(build_dir), "--target", "gemm_runner", "-j", str(args.jobs)]
    configure_code, configure_log, configure_time = run_command(configure, timeout=args.timeout)
    build_code = 1
    build_log = ""
    build_time = 0.0
    if configure_code == 0:
        build_code, build_log, build_time = run_command(build, timeout=args.timeout)
    full_log = "$ " + " ".join(configure) + "\n" + configure_log + "\n$ " + " ".join(build) + "\n" + build_log
    write_text(log_path, full_log)
    success = configure_code == 0 and build_code == 0 and binary_path.exists()
    return {
        "run_id": args.run_id,
        "task_id": task_id,
        "prompt_id": prompt_id,
        "sample_id": sample_id,
        "target_gpu": target_gpu,
        "target_arch": args.arch,
        "compile_success": success,
        "compile_time_sec": round(configure_time + build_time, 4),
        "binary_path": rel(binary_path) if success else "",
        "compile_log_path": rel(log_path),
        "error_type": "" if success else classify_compile_error(full_log),
        "error_summary": "" if success else summarize_error(full_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build generated kernels against the CUDA harness.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--target_gpu", required=True)
    parser.add_argument("--arch", required=True, help="sm80, sm90, sm70, sm89, or CMake numeric form.")
    parser.add_argument("--task")
    parser.add_argument("--prompt")
    parser.add_argument("--sample")
    parser.add_argument("--kernel_source", help="Direct kernel source for smoke/reference builds.")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    ensure_run_dirs(args.run_id)
    target_gpu = normalize_target(args.target_gpu)
    rows = []
    if args.kernel_source:
        kernel_source = (ROOT / args.kernel_source).resolve() if not args.kernel_source.startswith("/") else args.kernel_source
        from pathlib import Path

        kernel_source = Path(kernel_source)
        task_id = args.task or "direct_kernel"
        prompt_id = args.prompt or "manual"
        sample_id = args.sample or "sample_00"
        sample_dir = ROOT / "generated" / args.run_id / task_id / prompt_id / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(kernel_source, sample_dir / "kernel_extracted.cu")
        update_json(
            sample_dir / "metadata.json",
            {
                "run_id": args.run_id,
                "task_id": task_id,
                "prompt_id": prompt_id,
                "sample_id": sample_id,
                "target_gpu": target_gpu,
                "target_arch": args.arch,
                "kernel_source": rel(kernel_source),
                "manual_core_patch": False,
                "wrapper_only": False,
            },
        )
        rows.append(build_one(args, task_id, prompt_id, sample_id, target_gpu, kernel_source))
    else:
        cases = iter_cases(
            args.run_id,
            task_filter=filter_csv(args.task),
            prompt_filter=filter_csv(args.prompt),
            sample_filter=filter_csv(args.sample),
            target_filter=target_gpu,
        )
        for case in cases:
            kernel_source = case.sample_dir / "kernel_extracted.cu"
            if not kernel_source.exists():
                rows.append(
                    {
                        "run_id": args.run_id,
                        "task_id": case.task_id,
                        "prompt_id": case.prompt_id,
                        "sample_id": case.sample_id,
                        "target_gpu": target_gpu,
                        "target_arch": args.arch,
                        "compile_success": False,
                        "compile_time_sec": 0,
                        "binary_path": "",
                        "compile_log_path": "",
                        "error_type": "missing_source",
                        "error_summary": "kernel_extracted.cu not found",
                    }
                )
                continue
            rows.append(build_one(args, case.task_id, case.prompt_id, case.sample_id, target_gpu, kernel_source))

    append_csv(ROOT / "results" / args.run_id / "compile_results.csv", FIELDNAMES, rows)
    print(f"recorded {len(rows)} compile results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
