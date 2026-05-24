#!/usr/bin/env python3

from __future__ import annotations

import argparse

from common import choose_code_block, filter_csv, iter_cases, read_text, update_json, write_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract CUDA code from raw LLM responses.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--tasks")
    parser.add_argument("--prompts")
    parser.add_argument("--samples")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cases = iter_cases(
        args.run_id,
        task_filter=filter_csv(args.tasks),
        prompt_filter=filter_csv(args.prompts),
        sample_filter=filter_csv(args.samples),
    )
    written = 0
    for case in cases:
        response_path = case.sample_dir / "response_raw.md"
        output_path = case.sample_dir / "kernel_extracted.cu"
        if output_path.exists() and not args.overwrite:
            continue
        if not response_path.exists() or not response_path.read_text(encoding="utf-8").strip():
            update_json(case.sample_dir / "metadata.json", {"extraction_status": "missing_response"})
            continue
        code = choose_code_block(read_text(response_path))
        write_text(output_path, code.rstrip() + "\n")
        write_text(
            case.sample_dir / "launch_extracted.cu",
            "#include <cuda_fp16.h>\n#include <cuda_runtime.h>\n\n"
            "extern \"C\" void launch_gemm_kernel(const half* A, const half* B, half* C, "
            "int M, int N, int K, cudaStream_t stream);\n",
        )
        update_json(
            case.sample_dir / "metadata.json",
            {
                "extraction_status": "extracted",
                "has_required_launcher": "launch_gemm_kernel" in code,
            },
        )
        written += 1
    print(f"extracted {written} kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
