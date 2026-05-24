import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_TOP_DIR = Path(__file__).resolve().parents[1]


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    json_path = run_dir / "iteration_results.json"
    jsonl_path = run_dir / "iteration_results.jsonl"
    if json_path.exists():
        return json.loads(json_path.read_text())
    records = []
    with jsonl_path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def classify_operator(problem_name: str) -> str:
    name = problem_name.lower()
    if "matmul" in name or "matrix" in name or "gemm" in name:
        return "matmul"
    if "conv" in name:
        return "convolution"
    if "pool" in name:
        return "pooling"
    if "norm" in name:
        return "normalization"
    if any(token in name for token in ["relu", "sigmoid", "tanh", "gelu", "selu", "elu", "softplus", "softmax", "swish"]):
        return "activation"
    if any(token in name for token in ["sum", "mean", "max", "min", "prod", "argmax", "argmin"]):
        return "reduction"
    return "other"


def finite_or_blank(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return str(value)


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("eval_result") or {}
    stats = result.get("runtime_stats") or {}
    metadata = result.get("metadata") or {}
    failure_mode = classify_failure_mode(result, metadata)
    return {
        "run_name": record.get("run_name"),
        "level": record.get("level"),
        "problem_id": record.get("problem_id"),
        "problem_name": record.get("problem_name"),
        "operator_type": classify_operator(record.get("problem_name", "")),
        "iteration": record.get("iteration"),
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "runtime_us": result.get("runtime"),
        "ref_runtime_us": result.get("ref_runtime"),
        "speedup": record.get("speedup"),
        "best_speedup_so_far": None,
        "runtime_std_us": stats.get("std"),
        "runtime_min_us": stats.get("min"),
        "runtime_max_us": stats.get("max"),
        "elapsed_sec": record.get("elapsed_sec"),
        "kernel_path": record.get("kernel_path"),
        "error_keys": ";".join(sorted(metadata.keys())),
        "failure_mode": failure_mode,
    }


def classify_failure_mode(result: dict[str, Any], metadata: dict[str, Any]) -> str:
    if result.get("correctness"):
        return ""
    if "generation_error" in metadata:
        return "generation_error"
    static_text = " ".join(str(item) for item in metadata.get("static_check_error", []))
    warning_text = " ".join(str(item) for item in metadata.get("static_check_warnings", []))
    combined_static = f"{static_text} {warning_text}".lower()
    if "static_check_error" in metadata:
        if "missing __global__" in combined_static or "torch" in combined_static or "aten" in combined_static:
            return "constraint_violation"
        return "static_check_failure"
    if "compile_error" in metadata:
        return "compile_error"
    if "runtime_error" in metadata or "runtime_error_name" in metadata:
        return "runtime_error"
    if "error_during_performance" in metadata:
        return "performance_error"
    if "eval_exception" in metadata:
        return "eval_exception"
    if result.get("compiled") and not result.get("correctness"):
        return "wrong_answer"
    return "unknown_failure"


def sign(value: float, eps: float) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def summarize_problem(rows: list[dict[str, Any]], eps: float) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: row["iteration"])
    correct_rows = [row for row in rows if row["correctness"] and row["speedup"] is not None]
    speedups = [row["speedup"] for row in correct_rows]
    first_correct_iter = correct_rows[0]["iteration"] if correct_rows else None
    initial_speedup = speedups[0] if speedups else None
    best_row = max(correct_rows, key=lambda row: row["speedup"], default=None)
    best_speedup = best_row["speedup"] if best_row else None
    best_iter = best_row["iteration"] if best_row else None
    final_correct_speedup = speedups[-1] if speedups else None

    best_so_far = []
    current_best = None
    for row in rows:
        if row["speedup"] is not None:
            current_best = row["speedup"] if current_best is None else max(current_best, row["speedup"])
        best_so_far.append(current_best)

    convergence_iter = None
    if best_speedup is not None:
        threshold = 0.95 * best_speedup
        for row, best_value in zip(rows, best_so_far):
            if best_value is not None and best_value >= threshold:
                convergence_iter = row["iteration"]
                break

    deltas = [speedups[i] - speedups[i - 1] for i in range(1, len(speedups))]
    signed_deltas = [sign(delta, eps) for delta in deltas if sign(delta, eps) != 0]
    oscillation_count = sum(
        1 for i in range(1, len(signed_deltas)) if signed_deltas[i] != signed_deltas[i - 1]
    )
    regression_count = sum(1 for delta in deltas if delta < -eps)

    relative_gain = None
    if initial_speedup is not None and best_speedup is not None and abs(initial_speedup) > eps:
        relative_gain = (best_speedup - initial_speedup) / abs(initial_speedup)

    if not correct_rows:
        category = "no-correct-kernel"
    elif relative_gain is not None and relative_gain >= 0.10 and best_iter != first_correct_iter:
        category = "iteration-dependent"
    elif regression_count > 0 or oscillation_count > 0:
        category = "oscillatory"
    elif relative_gain is not None and relative_gain < 0.05:
        category = "low-iteration-benefit"
    else:
        category = "early-converged"

    return {
        "level": rows[0]["level"],
        "problem_id": rows[0]["problem_id"],
        "problem_name": rows[0]["problem_name"],
        "operator_type": rows[0]["operator_type"],
        "num_iterations": len(rows),
        "compile_rate": sum(1 for row in rows if row["compiled"]) / len(rows),
        "correctness_rate": sum(1 for row in rows if row["correctness"]) / len(rows),
        "constraint_violation_count": sum(1 for row in rows if row.get("failure_mode") == "constraint_violation"),
        "first_correct_iter": first_correct_iter,
        "convergence_iter_95pct_best": convergence_iter,
        "best_iter": best_iter,
        "initial_correct_speedup": initial_speedup,
        "best_speedup": best_speedup,
        "final_correct_speedup": final_correct_speedup,
        "relative_gain_from_first_correct": relative_gain,
        "oscillation_count": oscillation_count,
        "regression_count": regression_count,
        "category": category,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_blank(row.get(key)) for key in keys})


def plot_lines(rows: list[dict[str, Any]], out_dir: Path, y_key: str, title: str, ylabel: str) -> None:
    by_problem: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem_id"]].append(row)

    plt.figure(figsize=(10, 6))
    for problem_id, problem_rows in sorted(by_problem.items()):
        problem_rows = sorted(problem_rows, key=lambda row: row["iteration"])
        xs = [row["iteration"] for row in problem_rows]
        ys = [row[y_key] if row[y_key] is not None else math.nan for row in problem_rows]
        label = f"P{problem_id} {problem_rows[0]['operator_type']}"
        plt.plot(xs, ys, marker="o", label=label)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{y_key}_by_iteration.png", dpi=160)
    plt.close()


def plot_summary(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [f"P{row['problem_id']}" for row in summary_rows]
    best = [row["best_speedup"] or 0 for row in summary_rows]
    oscillations = [row["oscillation_count"] for row in summary_rows]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, best)
    plt.title("Best Speedup By Problem")
    plt.xlabel("problem")
    plt.ylabel("best speedup")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "best_speedup_by_problem.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, oscillations)
    plt.title("Oscillation Count By Problem")
    plt.xlabel("problem")
    plt.ylabel("sign changes in speedup deltas")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "oscillation_count_by_problem.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze iterative KernelBench convergence runs.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = REPO_TOP_DIR / "runs" / args.run_name
    out_dir = Path(args.output_dir) if args.output_dir else REPO_TOP_DIR / "reports" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(run_dir)
    rows = [flatten_record(record) for record in records]

    best_by_problem: dict[int, float | None] = defaultdict(lambda: None)
    for row in sorted(rows, key=lambda row: (row["problem_id"], row["iteration"])):
        current = best_by_problem[row["problem_id"]]
        if row["speedup"] is not None:
            current = row["speedup"] if current is None else max(current, row["speedup"])
            best_by_problem[row["problem_id"]] = current
        row["best_speedup_so_far"] = current

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["problem_id"]].append(row)
    summary_rows = [summarize_problem(problem_rows, args.epsilon) for problem_rows in grouped.values()]
    summary_rows = sorted(summary_rows, key=lambda row: row["problem_id"])

    write_csv(out_dir / "iteration_metrics.csv", rows)
    write_csv(out_dir / "convergence_summary.csv", summary_rows)
    plot_lines(rows, out_dir, "speedup", "Speedup By Iteration", "speedup vs reference")
    plot_lines(rows, out_dir, "best_speedup_so_far", "Best-So-Far Speedup By Iteration", "best speedup so far")
    plot_lines(rows, out_dir, "runtime_us", "Runtime By Iteration", "runtime (us)")
    plot_summary(summary_rows, out_dir)

    print(f"[done] wrote analysis to {out_dir}")
    for row in summary_rows:
        print(
            f"P{row['problem_id']} {row['operator_type']} category={row['category']} "
            f"best={row['best_speedup']} best_iter={row['best_iter']} oscillations={row['oscillation_count']}"
        )


if __name__ == "__main__":
    main()