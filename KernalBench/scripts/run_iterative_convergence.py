import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.prompt_constructor_toml import get_custom_prompt, get_prompt_for_backend
from kernelbench.utils import (
    SERVER_PRESETS,
    create_inference_server_from_presets,
    extract_first_code,
    set_gpu_arch,
)


REPO_TOP_DIR = Path(__file__).resolve().parents[1]


def parse_problem_ids(value: str) -> list[int]:
    problem_ids: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            problem_ids.extend(range(int(start), int(end) + 1))
        else:
            problem_ids.append(int(item))
    return sorted(dict.fromkeys(problem_ids))


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "dict"):
        data = result.dict()
    else:
        data = dict(result)
    return safe_json(data)


def make_prompt(args: argparse.Namespace, ref_arch_src: str) -> str:
    include_hardware = str(args.include_hardware_info).lower() in {"1", "true", "yes"}
    if args.custom_prompt_key:
        return get_custom_prompt(
            args.custom_prompt_key,
            ref_arch_src=ref_arch_src,
            backend=args.backend,
            option=args.prompt_option,
            precision=args.precision,
            include_hardware=include_hardware,
            gpu_name=args.hardware_gpu_name,
        )
    return get_prompt_for_backend(
        ref_arch_src,
        args.backend,
        option=args.prompt_option,
        precision=args.precision,
        include_hardware=include_hardware,
        gpu_name=args.hardware_gpu_name,
    )


def summarize_record(record: dict[str, Any]) -> str:
    result = record.get("eval_result") or {}
    metadata = result.get("metadata") or {}
    runtime = result.get("runtime", -1.0)
    ref_runtime = result.get("ref_runtime", -1.0)
    speedup = record.get("speedup")
    lines = [
        f"compiled: {result.get('compiled')}",
        f"correctness: {result.get('correctness')}",
        f"runtime_us: {runtime}",
        f"ref_runtime_us: {ref_runtime}",
        f"speedup: {speedup}",
    ]
    for key in [
        "static_check_error",
        "runtime_error_name",
        "runtime_error",
        "compile_error",
        "other_error",
        "error_during_performance",
        "generation_error",
        "eval_exception",
    ]:
        if key in metadata:
            lines.append(f"{key}: {metadata[key]}")
    return "\n".join(lines)


def build_feedback_prompt(
    base_prompt: str,
    problem_name: str,
    backend: str,
    iteration: int,
    previous_code: str,
    previous_record: dict[str, Any],
    best_record: dict[str, Any] | None,
) -> str:
    best_summary = "No correct kernel has been found yet."
    if best_record is not None:
        best_summary = summarize_record(best_record)

    return f"""{base_prompt}

You are now on optimization iteration {iteration} for KernelBench problem `{problem_name}`.
Use the previous attempt and evaluator feedback below to produce a better complete replacement.

Rules for this iteration:
- Return only one complete Python code block.
- The code must define `ModelNew` and be directly evaluable by KernelBench.
- Preserve numerical correctness before optimizing runtime.
- If the previous kernel failed to compile or failed correctness, fix that first.
- If it was correct, improve runtime on the same target GPU and input shapes.
- Do not replace the target operator with PyTorch/ATen/cuBLAS/cuDNN fallback calls such as `torch.matmul`, `torch.relu`, `torch.nn.functional`, `at::matmul`, or `at::batch_norm`.
- For CUDA backend answers, include a real custom `__global__` kernel in `cuda_sources`; wrapper-only extensions will fail the static checker.
- The active backend is `{backend}`; keep the implementation within that backend's KernelBench constraints.

Previous iteration feedback:
{summarize_record(previous_record)}

Best correct kernel so far:
{best_summary}

Previous kernel code:
```python
{previous_code}
```
"""


def compute_speedup(result: dict[str, Any]) -> float | None:
    runtime = result.get("runtime", -1.0)
    ref_runtime = result.get("ref_runtime", -1.0)
    if result.get("correctness") and runtime and runtime > 0 and ref_runtime and ref_runtime > 0:
        return ref_runtime / runtime
    return None


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(safe_json(record), sort_keys=True) + "\n")


def evaluate_kernel(
    args: argparse.Namespace,
    ref_arch_src: str,
    kernel_src: str,
    build_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], str | None]:
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        static_ok, static_errors, static_warnings = validate_kernel_static(
            kernel_src,
            backend=args.backend,
            precision=args.precision,
        )
        if not static_ok and args.check_kernel:
            return {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "runtime_stats": {},
                "ref_runtime": -1.0,
                "ref_runtime_stats": {},
                "metadata": {
                    "static_check_error": static_errors,
                    "static_check_warnings": static_warnings,
                },
            }, None
    except Exception as exc:
        if args.check_kernel:
            return {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "runtime_stats": {},
                "ref_runtime": -1.0,
                "ref_runtime_stats": {},
                "metadata": {"static_check_exception": str(exc)},
            }, None

    if args.isolate_eval:
        ref_path = build_dir / "reference.py"
        kernel_path = build_dir / "kernel.py"
        request_path = build_dir / "eval_request.json"
        output_path = build_dir / "eval_result.json"
        stdout_path = build_dir / "eval_stdout.txt"
        stderr_path = build_dir / "eval_stderr.txt"
        ref_path.write_text(ref_arch_src)
        kernel_path.write_text(kernel_src)
        request = {
            "ref_path": str(ref_path),
            "kernel_path": str(kernel_path),
            "build_dir": str(build_dir / "extension_build"),
            "device": str(device),
            "backend": args.backend,
            "precision": args.precision,
            "num_correct_trials": args.num_correct_trials,
            "num_perf_trials": args.num_perf_trials,
            "timing_method": args.timing_method,
            "verbose": args.verbose,
        }
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True))
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_TOP_DIR / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_TOP_DIR / "scripts" / "evaluate_kernel_once.py"),
                    "--request-path",
                    str(request_path),
                    "--output-path",
                    str(output_path),
                ],
                cwd=REPO_TOP_DIR,
                env=env,
                text=True,
                capture_output=True,
                timeout=args.eval_timeout_sec,
                check=False,
            )
            stdout_path.write_text(completed.stdout or "")
            stderr_path.write_text(completed.stderr or "")
            if output_path.exists():
                payload = json.loads(output_path.read_text())
                result = payload.get("result") or {}
                metadata = result.setdefault("metadata", {})
                if completed.returncode != 0:
                    metadata["subprocess_returncode"] = completed.returncode
                    metadata["subprocess_stderr_path"] = str(stderr_path.relative_to(REPO_TOP_DIR))
                return safe_json(result), None
            return {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "runtime_stats": {},
                "ref_runtime": -1.0,
                "ref_runtime_stats": {},
                "metadata": {
                    "eval_exception": "isolated evaluator produced no output JSON",
                    "subprocess_returncode": completed.returncode,
                    "subprocess_stdout_path": str(stdout_path.relative_to(REPO_TOP_DIR)),
                    "subprocess_stderr_path": str(stderr_path.relative_to(REPO_TOP_DIR)),
                },
            }, None
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "")
            stderr_path.write_text(exc.stderr or "")
            return {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "runtime_stats": {},
                "ref_runtime": -1.0,
                "ref_runtime_stats": {},
                "metadata": {
                    "eval_exception": f"isolated evaluator timed out after {args.eval_timeout_sec} seconds",
                    "subprocess_stdout_path": str(stdout_path.relative_to(REPO_TOP_DIR)),
                    "subprocess_stderr_path": str(stderr_path.relative_to(REPO_TOP_DIR)),
                },
            }, None

    try:
        result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            measure_performance=True,
            timing_method=args.timing_method,
            verbose=args.verbose,
            build_dir=build_dir,
            device=device,
            backend=args.backend,
            precision=get_torch_dtype_from_string(args.precision),
            check_for_excessive_speedup=True,
        )
        return result_to_dict(result), None
    except Exception:
        return {
            "compiled": False,
            "correctness": False,
            "runtime": -1.0,
            "runtime_stats": {},
            "ref_runtime": -1.0,
            "ref_runtime_stats": {},
            "metadata": {
                "eval_exception": traceback.format_exc(limit=20),
            },
        }, None


def run_problem(
    args: argparse.Namespace,
    dataset: Any,
    inference_server: Any,
    run_dir: Path,
    results_path: Path,
    problem_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    problem = dataset.get_problem_by_id(problem_id)
    problem_name = problem.name
    ref_arch_src = problem.code
    base_prompt = make_prompt(args, ref_arch_src)

    previous_code = ""
    previous_record: dict[str, Any] | None = None
    best_record: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []

    for iteration in range(args.num_iterations):
        prompt = base_prompt
        if iteration > 0 and previous_record is not None:
            prompt = build_feedback_prompt(
                base_prompt,
                problem_name,
                args.backend,
                iteration,
                previous_code,
                previous_record,
                best_record,
            )

        stem = f"level_{args.level}_problem_{problem_id}_iter_{iteration}"
        prompt_path = run_dir / f"{stem}_prompt.txt"
        raw_path = run_dir / f"{stem}_raw.txt"
        kernel_path = run_dir / f"{stem}_kernel.py"
        prompt_path.write_text(prompt)

        started_at = time.time()
        generation_error = None
        raw_output = ""
        kernel_src = ""
        try:
            raw_output = inference_server(prompt)
            raw_path.write_text(raw_output or "")
            kernel_src = extract_first_code(raw_output, ["python", "cpp"]) or (raw_output or "").strip()
            kernel_path.write_text(kernel_src)
        except Exception:
            generation_error = traceback.format_exc(limit=20)
            raw_path.write_text(raw_output or "")

        if generation_error or not kernel_src:
            eval_result = {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "runtime_stats": {},
                "ref_runtime": -1.0,
                "ref_runtime_stats": {},
                "metadata": {"generation_error": generation_error or "empty_generation"},
            }
        else:
            build_dir = run_dir / "build" / f"level_{args.level}" / f"problem_{problem_id}" / f"iter_{iteration}"
            eval_result, _ = evaluate_kernel(args, ref_arch_src, kernel_src, build_dir, device)

        speedup = compute_speedup(eval_result)
        record = {
            "run_name": args.run_name,
            "level": args.level,
            "problem_id": problem_id,
            "problem_name": problem_name,
            "iteration": iteration,
            "backend": args.backend,
            "precision": args.precision,
            "server_type": args.server_type,
            "model_name": args.model_name,
            "prompt_path": str(prompt_path.relative_to(REPO_TOP_DIR)),
            "raw_path": str(raw_path.relative_to(REPO_TOP_DIR)),
            "kernel_path": str(kernel_path.relative_to(REPO_TOP_DIR)),
            "eval_result": eval_result,
            "speedup": speedup,
            "elapsed_sec": time.time() - started_at,
        }

        records.append(record)
        append_jsonl(results_path, record)

        if speedup is not None and (best_record is None or speedup > (best_record.get("speedup") or -1.0)):
            best_record = record
        previous_record = record
        previous_code = kernel_src

        status = "correct" if eval_result.get("correctness") else "failed"
        print(
            f"[iter] level={args.level} problem={problem_id} iter={iteration} "
            f"status={status} speedup={speedup} runtime={eval_result.get('runtime')}"
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative KernelBench generation/evaluation experiments.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset-src", default="local", choices=["local", "huggingface"])
    parser.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problem-ids", required=True, help="Comma-separated IDs and ranges, e.g. 1,6,19 or 1-5")
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--server-type", default="azure")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--backend", default="cuda", choices=["cuda", "triton", "cute", "tilelang", "thunderkittens"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--prompt-option", default="one_shot", choices=["zero_shot", "one_shot", "few_shot"])
    parser.add_argument("--custom-prompt-key", default=None)
    parser.add_argument("--include-hardware-info", action="store_true")
    parser.add_argument("--hardware-gpu-name", default=None)
    parser.add_argument("--gpu-arch", default="Hopper")
    parser.add_argument("--num-correct-trials", type=int, default=3)
    parser.add_argument("--num-perf-trials", type=int, default=30)
    parser.add_argument("--timing-method", default="cuda_event")
    parser.add_argument("--isolate-eval", action="store_true")
    parser.add_argument("--eval-timeout-sec", type=int, default=600)
    parser.add_argument("--check-kernel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv(REPO_TOP_DIR / ".env")

    if args.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[args.server_type]
        args.model_name = args.model_name or preset.get("model_name")
        args.max_tokens = args.max_tokens or preset.get("max_tokens")
        args.temperature = args.temperature if args.temperature is not None else preset.get("temperature")

    run_dir = REPO_TOP_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "iteration_results.jsonl"
    if results_path.exists():
        results_path.unlink()

    metadata = vars(args).copy()
    metadata["problem_ids_expanded"] = parse_problem_ids(args.problem_ids)
    metadata["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "iteration_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    set_gpu_arch([args.gpu_arch])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    inference_server = create_inference_server_from_presets(
        server_type=args.server_type,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        time_generation=True,
    )
    dataset = construct_kernelbench_dataset(
        level=args.level,
        source=args.dataset_src,
        dataset_name=args.dataset_name,
    )

    all_records: list[dict[str, Any]] = []
    for problem_id in parse_problem_ids(args.problem_ids):
        all_records.extend(run_problem(args, dataset, inference_server, run_dir, results_path, problem_id, device))

    (run_dir / "iteration_results.json").write_text(json.dumps(safe_json(all_records), indent=2, sort_keys=True))
    print(f"[done] wrote {len(all_records)} records to {run_dir}")


if __name__ == "__main__":
    main()