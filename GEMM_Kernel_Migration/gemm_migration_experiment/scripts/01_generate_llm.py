#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from common import (
    ROOT,
    abs_path,
    ensure_run_dirs,
    filter_csv,
    iter_cases,
    read_data,
    read_text,
    update_json,
    write_text,
)


def load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = abs_path(path)
    if not env_path.exists():
        raise RuntimeError(f"env file not found: {env_path}")
    for raw_line in read_text(env_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_prompt(case) -> str:
    metadata = case.metadata
    prompt_template = read_text(ROOT / "prompts" / f"{case.prompt_id}.md")
    source_kernel = read_text(abs_path(metadata["source_kernel_path"]))
    target_key = metadata.get("target_key", "")
    target_ref_dir = {
        "v100": "references/v100_sm70",
        "a100": "references/a100_sm80",
        "h100": "references/h100_sm90",
    }.get(target_key, "")
    target_notes = ""
    target_example = ""
    if target_ref_dir:
        notes_path = ROOT / target_ref_dir / "notes.md"
        kernel_path = ROOT / target_ref_dir / "kernel.cu"
        if notes_path.exists():
            target_notes = read_text(notes_path)
        if kernel_path.exists():
            target_example = read_text(kernel_path)
    return prompt_template.format(
        source_gpu=metadata.get("source_gpu", ""),
        target_gpu=metadata.get("target_gpu", ""),
        source_arch=metadata.get("source_arch", ""),
        target_arch=metadata.get("target_arch", ""),
        source_kernel=source_kernel,
        target_notes=target_notes,
        target_example=target_example,
    )


def call_litellm(model: str, prompt: str, temperature: float | None, max_tokens: int) -> str:
    try:
        from litellm import completion  # type: ignore
    except Exception as error:
        if model.startswith("azure/"):
            return call_azure_openai(model, prompt, temperature, max_tokens)
        raise RuntimeError("litellm is not installed; use --dry_run or install litellm") from error

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        if not (model.startswith("azure/") and temperature != 1.0):
            kwargs["temperature"] = temperature
    response = completion(**kwargs)
    return response.choices[0].message.content or ""


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _azure_url(base: str, deployment: str, api_version: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/openai"):
        root = base
    elif "/openai/deployments/" in base:
        return f"{base}/chat/completions?api-version={api_version}"
    else:
        root = f"{base}/openai"
    return f"{root}/deployments/{deployment}/chat/completions?api-version={api_version}"


def _post_json(url: str, api_key: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Azure OpenAI HTTP {error.code}: {body[:1000]}") from error


def call_azure_openai(model: str, prompt: str, temperature: float | None, max_tokens: int) -> str:
    api_key = _env_first("AZURE_API_KEY", "AZURE_OPENAI_API_KEY")
    base = _env_first("AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_BASE")
    api_version = _env_first("AZURE_API_VERSION", "AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"
    deployment = ""
    if model.startswith("azure/"):
        deployment = model.split("/", 1)[1]
    if not deployment:
        deployment = _env_first("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_DEPLOYMENT_NAME")
    if not api_key or not base or not deployment:
        raise RuntimeError("missing Azure OpenAI env vars: need key, base endpoint, and deployment")

    url = _azure_url(base, deployment, api_version)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None and not (model.startswith("azure/") and temperature != 1.0):
        payload["temperature"] = temperature

    try:
        response = _post_json(url, api_key, payload)
    except RuntimeError as error:
        message = str(error)
        retry_payload = dict(payload)
        changed = False
        if "max_tokens" in message and "max_completion_tokens" in message:
            retry_payload.pop("max_tokens", None)
            retry_payload["max_completion_tokens"] = max_tokens
            changed = True
        if "temperature" in message and "unsupported" in message.lower():
            retry_payload.pop("temperature", None)
            changed = True
        if not changed:
            raise
        response = _post_json(url, api_key, retry_payload)

    return response["choices"][0]["message"].get("content") or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LLM responses for prepared cases.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model", help="Model name. Defaults to configs/models.yaml.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--env_file", help="Optional .env file to load without printing secrets.")
    parser.add_argument("--tasks")
    parser.add_argument("--prompts")
    parser.add_argument("--samples")
    parser.add_argument("--all_cases", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Write prompt_preview.md without calling an API.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_run_dirs(args.run_id)
    load_env_file(args.env_file)
    model_config = read_data(ROOT / "configs" / "models.yaml")
    model = args.model or model_config.get("default_model")
    temperature = args.temperature if args.temperature is not None else model_config.get("temperature")
    max_tokens = args.max_tokens or int(model_config.get("max_tokens", 16384))

    cases = iter_cases(
        args.run_id,
        task_filter=filter_csv(args.tasks),
        prompt_filter=filter_csv(args.prompts),
        sample_filter=filter_csv(args.samples),
    )
    if not cases:
        raise SystemExit(f"no prepared cases found for run_id={args.run_id}")
    if not args.all_cases and not (args.tasks or args.prompts or args.samples):
        raise SystemExit("refusing to generate all cases without --all_cases")

    completed = 0
    for case in cases:
        response_path = case.sample_dir / "response_raw.md"
        if response_path.exists() and response_path.read_text(encoding="utf-8").strip() and not args.overwrite:
            continue
        prompt = build_prompt(case)
        write_text(case.sample_dir / "prompt_preview.md", prompt)
        if args.dry_run:
            update_json(case.sample_dir / "metadata.json", {"response_status": "dry_run_prompt_only"})
            completed += 1
            continue
        response_text = call_litellm(model, prompt, temperature, max_tokens)
        write_text(response_path, response_text)
        update_json(
            case.sample_dir / "metadata.json",
            {
                "model": model,
                "temperature": temperature,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "response_status": "generated",
            },
        )
        completed += 1

    print(f"processed {completed} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
