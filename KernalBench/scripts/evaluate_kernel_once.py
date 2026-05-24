import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one KernelBench candidate in an isolated process.")
    parser.add_argument("--request-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request_path).read_text())
    output_path = Path(args.output_path)
    device = torch.device(request["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)

    try:
        result = eval_kernel_against_ref(
            original_model_src=Path(request["ref_path"]).read_text(),
            custom_model_src=Path(request["kernel_path"]).read_text(),
            num_correct_trials=request["num_correct_trials"],
            num_perf_trials=request["num_perf_trials"],
            measure_performance=True,
            timing_method=request["timing_method"],
            verbose=request["verbose"],
            build_dir=request["build_dir"],
            device=device,
            backend=request["backend"],
            precision=get_torch_dtype_from_string(request["precision"]),
            check_for_excessive_speedup=True,
        )
        payload = {"result": result_to_dict(result)}
    except Exception:
        payload = {
            "result": {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "runtime_stats": {},
                "ref_runtime": -1.0,
                "ref_runtime_stats": {},
                "metadata": {"eval_exception": traceback.format_exc(limit=20)},
            }
        }

    output_path.write_text(json.dumps(safe_json(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()