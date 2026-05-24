#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_data(path: Path | str) -> Any:
    path = Path(path)
    text = read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as error:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                f"{path} is not JSON and PyYAML is not installed"
            ) from error
        return yaml.safe_load(text)


def write_json(path: Path | str, data: Any) -> None:
    write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def update_json(path: Path | str, updates: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(read_text(path))
    data.update(updates)
    write_json(path, data)
    return data


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def abs_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def ensure_run_dirs(run_id: str) -> None:
    for name in ["generated", "build", "logs", "results", "reports"]:
        (ROOT / name / run_id).mkdir(parents=True, exist_ok=True)
    for subdir in ["compile", "correctness", "correctness_audit", "performance", "ncu", "llm", "sass"]:
        (ROOT / "logs" / run_id / subdir).mkdir(parents=True, exist_ok=True)


def load_matrix() -> dict[str, Any]:
    return read_data(ROOT / "configs" / "experiment_matrix.yaml")


def load_tasks() -> list[dict[str, Any]]:
    return list(load_matrix().get("required_tasks", []))


def load_prompt_ids() -> list[str]:
    return list(load_matrix().get("prompts", []))


def filter_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def cmake_arch(arch: str) -> str:
    arch = arch.strip().lower().replace("sm_", "sm")
    if arch.startswith("sm"):
        return arch[2:]
    return arch


def normalize_target(value: str) -> str:
    value = value.strip().lower()
    if value in {"nvidia a100", "a100", "sm80"}:
        return "a100"
    if value in {"nvidia h100", "h100", "sm90"}:
        return "h100"
    if value in {"nvidia v100", "v100", "sm70"}:
        return "v100"
    return value


def append_csv(path: Path | str, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class CaseRef:
    run_id: str
    task_id: str
    prompt_id: str
    sample_id: str
    sample_dir: Path
    metadata: dict[str, Any]


def iter_cases(
    run_id: str,
    task_filter: set[str] | None = None,
    prompt_filter: set[str] | None = None,
    sample_filter: set[str] | None = None,
    target_filter: str | None = None,
) -> list[CaseRef]:
    generated = ROOT / "generated" / run_id
    cases: list[CaseRef] = []
    if not generated.exists():
        return cases
    target_filter = normalize_target(target_filter) if target_filter else None
    for task_dir in sorted(path for path in generated.iterdir() if path.is_dir()):
        if task_filter and task_dir.name not in task_filter:
            continue
        for prompt_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            if prompt_filter and prompt_dir.name not in prompt_filter:
                continue
            for sample_dir in sorted(path for path in prompt_dir.iterdir() if path.is_dir()):
                if sample_filter and sample_dir.name not in sample_filter:
                    continue
                metadata_path = sample_dir / "metadata.json"
                metadata: dict[str, Any] = {}
                if metadata_path.exists():
                    metadata = json.loads(read_text(metadata_path))
                if target_filter:
                    sample_target = normalize_target(
                        str(metadata.get("target_key") or metadata.get("target_gpu") or "")
                    )
                    if sample_target != target_filter:
                        continue
                cases.append(
                    CaseRef(
                        run_id=run_id,
                        task_id=task_dir.name,
                        prompt_id=prompt_dir.name,
                        sample_id=sample_dir.name,
                        sample_dir=sample_dir,
                        metadata=metadata,
                    )
                )
    return cases


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, float]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    start = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout, time.time() - start
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + (error.stderr or "")
        return 124, output + "\nTIMEOUT\n", time.time() - start


def classify_compile_error(log: str) -> str:
    lower = log.lower()
    if "unsupported" in lower or "requires .target sm_" in lower:
        return "unsupported_intrinsic"
    if "identifier" in lower and "undefined" in lower:
        return "missing_header"
    if "no instance of overloaded function" in lower or "cannot convert" in lower:
        return "type_error"
    if "error:" in lower and ("expected" in lower or "syntax" in lower):
        return "syntax_error"
    if "undefined reference" in lower:
        return "link_error"
    if "sm_" in lower and "not defined" in lower:
        return "wrong_arch"
    if "template" in lower:
        return "template_error"
    if "error" in lower or "failed" in lower:
        return "unknown"
    return ""


def summarize_error(log: str, limit: int = 500) -> str:
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if any(token in line.lower() for token in ["error", "unsupported", "failed", "undefined"])
    ]
    summary = " | ".join(interesting[:5] or lines[-5:])
    return summary[:limit]


def extract_code_fences(markdown: str) -> list[str]:
    fence_re = re.compile(r"```(?:cuda|cu|cpp|c\+\+|cc|cxx)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    return [match.group(1).strip() for match in fence_re.finditer(markdown)]


def choose_code_block(markdown: str) -> str:
    blocks = extract_code_fences(markdown)
    if not blocks:
        return markdown.strip()
    launch_blocks = [block for block in blocks if "launch_gemm_kernel" in block]
    if launch_blocks:
        return max(launch_blocks, key=len)
    return max(blocks, key=len)


def case_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("run_id", "")),
        str(row.get("task_id", "")),
        str(row.get("prompt_id", "")),
        str(row.get("sample_id", "")),
    )
