#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import contextlib
import importlib
import io
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4


SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = SKILL_ROOT / "logs"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "runs" / "agent"
MODULE_CACHE: dict[str, Any] = {}
ULTRALYTICS_INIT = REPO_ROOT / "ultralytics" / "__init__.py"
DEFAULT_CFG_FILE = REPO_ROOT / "ultralytics" / "cfg" / "default.yaml"
DATASET_CFG_DIR = REPO_ROOT / "ultralytics" / "cfg" / "datasets"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RUNTIME_CACHE_FILE = LOG_DIR / "runtime-cache.json"
RUNTIME_CACHE_TTL_SEC = 600
IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
MULTIMODAL_PARAM_KEYS = {
    "prompt",
    "question",
    "system_prompt",
    "developer_prompt",
    "thinking_with_image",
    "method",
    "provider",
    "vlm_provider",
    "vlm_model",
    "llm_model",
    "openai_base_url",
    "openai_api_mode",
    "image_detail",
    "max_output_tokens",
    "temperature",
    "max_reasoning_items",
    "max_reasoning_boxes",
    "max_image_bytes",
    "structured_output",
    "enable_llm_refine",
    "skip_yolo",
    "detections",
}
MULTIMODAL_EVALUATE_PARAM_KEYS = {
    "data",
    "split",
    "limit",
    "max_images",
    "offset",
    "stride",
    "shuffle",
    "seed",
    "prompt_template",
    "include_ground_truth",
    "include_ground_truth_in_prompt",
    "run_yolo_val",
    "continue_on_error",
    "report_name",
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)


def cached(name: str, loader):
    if name not in MODULE_CACHE:
        MODULE_CACHE[name] = loader()
    return MODULE_CACHE[name]


def get_ultralytics_core() -> dict[str, Any]:
    def _loader():
        ultralytics = importlib.import_module("ultralytics")
        utils = importlib.import_module("ultralytics.utils")
        return {
            "YOLO": ultralytics.YOLO,
            "version": ultralytics.__version__,
            "SETTINGS": utils.SETTINGS,
            "SETTINGS_FILE": utils.SETTINGS_FILE,
            "YAML": utils.YAML,
        }

    return cached("ultralytics_core", _loader)


def get_cfg_helpers() -> dict[str, Any]:
    def _loader():
        cfg = importlib.import_module("ultralytics.cfg")
        return {
            "DEFAULT_CFG_PATH": cfg.DEFAULT_CFG_PATH,
            "copy_default_cfg": cfg.copy_default_cfg,
            "handle_yolo_solutions": cfg.handle_yolo_solutions,
        }

    return cached("cfg_helpers", _loader)


def get_checks_helpers() -> dict[str, Any]:
    def _loader():
        checks = importlib.import_module("ultralytics.utils.checks")
        return {"collect_system_info": checks.collect_system_info}

    return cached("checks_helpers", _loader)


def get_moe_helpers() -> dict[str, Any]:
    def _loader():
        analysis = importlib.import_module("ultralytics.nn.modules.moe.analysis")
        pruning = importlib.import_module("ultralytics.nn.modules.moe.pruning")
        return {
            "diagnose_model": analysis.diagnose_model,
            "prune_moe_model": pruning.prune_moe_model,
        }

    return cached("moe_helpers", _loader)


def get_ultralytics_module_info() -> dict[str, Any]:
    def _loader():
        spec = importlib.util.find_spec("ultralytics")
        module_path = Path(spec.origin).resolve() if spec and spec.origin else ULTRALYTICS_INIT.resolve()
        return {
            "path": str(module_path),
            "version": read_repo_version(),
            "local_repo_active": REPO_ROOT in module_path.parents or module_path == ULTRALYTICS_INIT.resolve(),
        }

    return cached("ultralytics_module_info", _loader)


def runtime_cache_enabled() -> bool:
    return os.environ.get("YOLO_MASTER_AGENT_RUNTIME_CACHE", "").lower() in {"1", "true", "yes"}


def read_torch_runtime_cache() -> dict[str, Any] | None:
    if not runtime_cache_enabled() or not RUNTIME_CACHE_FILE.exists():
        return None
    try:
        if time.time() - RUNTIME_CACHE_FILE.stat().st_mtime > RUNTIME_CACHE_TTL_SEC:
            return None
        payload = json.loads(RUNTIME_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("python") != sys.executable or payload.get("platform") != sys.platform:
        return None
    data = payload.get("torch")
    return data if isinstance(data, dict) else None


def write_torch_runtime_cache(info: dict[str, Any]) -> None:
    if not runtime_cache_enabled():
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        RUNTIME_CACHE_FILE.write_text(
            json.dumps(
                {"python": sys.executable, "platform": sys.platform, "torch": json_safe(info)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def get_torch_runtime() -> dict[str, Any]:
    def _loader():
        cached_info = read_torch_runtime_cache()
        if cached_info is not None:
            return cached_info
        info: dict[str, Any] = {
            "installed": False,
            "version": None,
            "cuda": {"available": False, "device_count": 0, "devices": []},
            "mps": {"built": False, "available": False},
        }
        try:
            torch = importlib.import_module("torch")
        except Exception:
            write_torch_runtime_cache(info)
            return info

        info["installed"] = True
        info["version"] = getattr(torch, "__version__", None)
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        cuda_devices = []
        cuda_count = 0
        if cuda_available:
            try:
                cuda_count = int(torch.cuda.device_count())
                cuda_devices = [torch.cuda.get_device_name(i) for i in range(cuda_count)]
            except Exception:
                cuda_count = 0
                cuda_devices = []
        info["cuda"] = {"available": cuda_available, "device_count": cuda_count, "devices": cuda_devices}
        try:
            info["mps"] = {
                "built": bool(torch.backends.mps.is_built()),
                "available": bool(torch.backends.mps.is_available()),
            }
        except Exception:
            info["mps"] = {"built": False, "available": False}
        write_torch_runtime_cache(info)
        return info

    return cached("torch_runtime", _loader)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "results_dict"):
        return json_safe(value.results_dict)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def read_repo_version() -> str:
    text = ULTRALYTICS_INIT.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError(f"Could not parse version from {ULTRALYTICS_INIT}")
    return match.group(1)


def read_default_cfg() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(DEFAULT_CFG_FILE.read_text(encoding="utf-8"))


def path_like(value: str) -> bool:
    return any(token in value for token in ("/", "\\", ".")) and not value.startswith(("http://", "https://"))


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, str) and path_like(value):
        p = Path(value)
        if not p.is_absolute():
            candidate = (REPO_ROOT / p).resolve()
            if candidate.exists() or value.startswith((".", "/")) or "/" in value:
                return str(candidate)
    if isinstance(value, str):
        builtin_dataset = (DATASET_CFG_DIR / value).resolve()
        if builtin_dataset.exists():
            return str(builtin_dataset)
    return value


def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    request = deepcopy(request)
    request.setdefault("workspace_root", str(REPO_ROOT))
    request.setdefault("request_id", default_request_id(request.get("skill", "skill")))
    request.setdefault("runtime", {})
    request.setdefault("inputs", {})
    request.setdefault("params", {})
    request.setdefault("artifacts", {})
    request.setdefault("policy", {})
    request["inputs"] = normalize_value(request["inputs"])
    request["params"] = normalize_value(request["params"])
    request["artifacts"] = normalize_value(request["artifacts"])
    return request


def is_dry_run(request: dict[str, Any]) -> bool:
    return bool(request.get("policy", {}).get("dry_run", False))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "skill"


def default_request_id(skill: str) -> str:
    return f"{slugify(skill)}-{uuid4().hex[:8]}"


def prefer_cli(request: dict[str, Any]) -> bool:
    runtime = request.get("runtime", {})
    if runtime.get("prefer_python_api"):
        return False
    return runtime.get("prefer_cli", True)


def mps_available() -> bool:
    return bool(get_torch_runtime()["mps"]["available"])


def available_devices() -> list[str]:
    torch_info = get_torch_runtime()
    devices = ["cpu"]
    if torch_info["mps"]["available"]:
        devices.insert(0, "mps")
    if torch_info["cuda"]["available"] and torch_info["cuda"]["device_count"] > 0:
        devices.insert(0, "cuda:0")
    return devices


def default_auto_device(request: dict[str, Any]) -> str | None:
    runtime = request.get("runtime", {})
    if not runtime.get("auto_detect_device", True):
        return None
    devices = available_devices()
    if runtime.get("prefer_mps", True) and "mps" in devices and sys.platform == "darwin":
        return "mps"
    if runtime.get("prefer_cuda", True) and "cuda:0" in devices:
        return "cuda:0"
    if runtime.get("prefer_mps", True) and "mps" in devices:
        return "mps"
    return devices[0] if devices else None


def resolve_device_selection(request: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    explicit = params.get("device")
    if explicit not in (None, "", "auto"):
        return {"device": str(explicit), "source": "params"}
    runtime = request.get("runtime", {})
    runtime_device = runtime.get("device")
    if runtime_device not in (None, "", "auto"):
        return {"device": str(runtime_device), "source": "runtime"}
    auto_device = default_auto_device(request)
    return {"device": auto_device, "source": "auto" if auto_device else None}


def resolve_default_device(request: dict[str, Any], params: dict[str, Any]) -> str | None:
    return resolve_device_selection(request, params)["device"]


def reference_state(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {"requested": value, "resolved": value, "exists": False}
    resolved = normalize_value(value)
    state = {"requested": value, "resolved": resolved}
    if isinstance(resolved, str) and path_like(resolved):
        state["exists"] = Path(resolved).exists()
    else:
        state["exists"] = bool(resolved)
    return state


def collect_environment_report(
    request: dict[str, Any],
    *,
    selected_device: str | None = None,
    requested_device: str | None = None,
    selection_source: str | None = None,
    cli_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = request.get("runtime", {})
    torch_info = get_torch_runtime()
    module_info = get_ultralytics_module_info()
    cli_path = None
    cli_status = "missing"
    install = None
    if cli_info:
        cli_path = cli_info.get("path")
        cli_status = cli_info.get("status", cli_status)
        install = cli_info
    else:
        cli_path = find_yolo_cli()
        cli_status = "available" if cli_path else "missing"
    selection = resolve_device_selection(request, request.get("params", {}))
    requested = requested_device if requested_device is not None else selection["device"]
    selected = selected_device if selected_device is not None else requested
    source = selection_source if selection_source is not None else selection["source"]
    report = {
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "workspace": {"repo_root": str(REPO_ROOT), "cwd": str(Path.cwd())},
        "ultralytics": {
            "repo_version": read_repo_version(),
            "module_version": module_info["version"],
            "module_path": module_info["path"],
            "local_repo_active": module_info["local_repo_active"],
        },
        "cli": {
            "available": bool(cli_path),
            "path": cli_path,
            "status": cli_status,
            "install": install,
        },
        "devices": {
            "requested": requested,
            "selected": selected,
            "available": available_devices(),
            "selection_source": source,
            "torch": torch_info,
        },
        "runtime": json_safe(runtime),
        "references": {
            "model": reference_state(request.get("inputs", {}).get("model")),
            "data": reference_state(request.get("inputs", {}).get("data") or request.get("params", {}).get("data")),
            "source": reference_state(request.get("inputs", {}).get("source")),
        },
    }
    return report


def doctor_recommendations(environment: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    cli = environment.get("cli", {})
    ultralytics = environment.get("ultralytics", {})
    devices = environment.get("devices", {})
    torch_info = devices.get("torch", {})

    if not cli.get("available"):
        recommendations.append("Run `python -m pip install -e .` to provision the local `yolo` CLI.")
    if not ultralytics.get("local_repo_active"):
        recommendations.append("Refresh the editable install with `python -m pip install -e .` so imports resolve to this repo.")
    if not torch_info.get("installed"):
        recommendations.append("Install PyTorch in the current environment before running train, val, benchmark, or predict.")
    if sys.platform == "darwin" and torch_info.get("mps", {}).get("available") and devices.get("selected") != "mps":
        recommendations.append("This host supports MPS; leave `device` unset or set `runtime.prefer_mps=true` for Apple Silicon acceleration.")
    if devices.get("selected") == "cpu" and sys.platform == "darwin" and not torch_info.get("mps", {}).get("available"):
        recommendations.append("MPS is unavailable, so heavy runs will stay on CPU until the PyTorch MPS runtime is available.")

    for label, state in (environment.get("references") or {}).items():
        if state.get("requested") not in (None, "") and not state.get("exists"):
            recommendations.append(f"Fix the `{label}` reference before launch: {state['requested']}")

    return recommendations


def apply_runtime_defaults(
    request: dict[str, Any],
    params: dict[str, Any],
    *,
    purpose: str,
) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    params = dict(params)
    auto_completed: dict[str, Any] = {}
    device_selection = resolve_device_selection(request, params)
    device = device_selection["device"]
    if device and "device" not in params:
        params["device"] = device
        auto_completed["device"] = device
        auto_completed["device_source"] = device_selection["source"]
    runtime = request.get("runtime", {})
    if purpose in {"train", "val", "benchmark"} and "workers" not in params and sys.platform == "darwin":
        params["workers"] = int(runtime.get("default_workers", 0))
        auto_completed["workers"] = params["workers"]
    return params, device, auto_completed


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def cli_install_command() -> list[str]:
    return [sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)]


def install_ultralytics_cli() -> dict[str, Any]:
    cmd = cli_install_command()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": strip_ansi(proc.stdout),
        "stderr": strip_ansi(proc.stderr),
    }


def find_yolo_cli() -> str | None:
    names = ("yolo", "yolo.exe", "yolo-script.py")
    candidates: list[Path] = []
    located = shutil.which("yolo")
    if located:
        candidates.append(Path(located))
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        base = Path(scripts_dir)
        candidates.extend(base / name for name in names)
    py_bin = Path(sys.executable).resolve().parent
    candidates.extend(py_bin / name for name in names)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and os.access(resolved, os.X_OK):
            return str(resolved)
    return None


def ensure_yolo_cli(force_install: bool = False) -> tuple[str, dict[str, Any]]:
    yolo_path = find_yolo_cli()
    if yolo_path and not force_install:
        return yolo_path, {"status": "available", "path": yolo_path}

    install = install_ultralytics_cli()
    yolo_path = find_yolo_cli()
    if install["returncode"] != 0 or not yolo_path:
        raise RuntimeError(
            "Failed to provision the `yolo` CLI via editable Ultralytics install.\n"
            f"cmd={install['cmd']}\nstdout={install['stdout']}\nstderr={install['stderr']}"
        )
    install["status"] = "installed"
    install["path"] = yolo_path
    return yolo_path, install


def cli_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    if value is None:
        return "None"
    return repr(value)


def kv_arg(key: str, value: Any) -> str:
    return f"{key}={cli_value(value)}"


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def repo_cli_env() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    prefix = str(REPO_ROOT)
    env["PYTHONPATH"] = prefix if not current else f"{prefix}{os.pathsep}{current}"
    return env


def cli_save_dir(request: dict[str, Any], params: dict[str, Any]) -> Path | None:
    project = params.get("project")
    name = params.get("name")
    if project and name:
        return resolved_path(str(project)) / str(name)
    if project:
        return resolved_path(str(project))
    return None


def inject_cli_artifact_location(request: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(params)
    if "project" not in enriched:
        enriched["project"] = str(DEFAULT_MANIFEST_DIR)
    if "name" not in enriched:
        enriched["name"] = request["request_id"]
    return enriched


def run_cli(args: list[str], cwd: Path | None = None, force_install: bool = False) -> dict[str, Any]:
    yolo_path, install = ensure_yolo_cli(force_install=force_install)
    cmd = [yolo_path, *args]
    proc = subprocess.run(cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True, env=repo_cli_env())
    return {
        "cmd": cmd,
        "cwd": str((cwd or REPO_ROOT).resolve()),
        "returncode": proc.returncode,
        "stdout": strip_ansi(proc.stdout),
        "stderr": strip_ansi(proc.stderr),
        "install": install,
    }


def cli_logs(cli_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "cmd": cli_result["cmd"],
        "cwd": cli_result["cwd"],
        "stdout": cli_result["stdout"],
        "stderr": cli_result["stderr"],
        "install": cli_result["install"],
    }


def cli_plan(
    request: dict[str, Any],
    args: list[str],
    cwd: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_extra = {"bootstrap": {"install_if_missing": cli_install_command()}}
    if extra:
        merged_extra.update(json_safe(extra))
    return plan_response(
        request,
        f"{request['skill']} CLI dry run prepared",
        "cli",
        "yolo",
        params={"cmd": ["yolo", *args], "cwd": str((cwd or REPO_ROOT).resolve())},
        extra=merged_extra,
    )


def coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except Exception:
            return text
    try:
        return float(text)
    except Exception:
        return text


def detect_cli_device(cli_result: dict[str, Any]) -> str | None:
    for arg in cli_result.get("cmd", [])[1:]:
        if isinstance(arg, str) and arg.startswith("device="):
            return arg.split("=", 1)[1]
    return None


def detect_missing_module(text: str) -> str | None:
    patterns = [
        re.compile(r"No module named ['\"]([^'\"]+)['\"]"),
        re.compile(r"No module named ([A-Za-z0-9_.-]+)"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def classify_cli_failure(cli_result: dict[str, Any]) -> dict[str, Any]:
    text = f"{cli_result.get('stdout', '')}\n{cli_result.get('stderr', '')}"
    info: dict[str, Any] = {
        "type": "CalledProcessError",
        "returncode": cli_result["returncode"],
        "device": detect_cli_device(cli_result),
    }
    hints: list[str] = []
    category = "cli_runtime_error"

    if "Expected more than 1 value per channel when training" in text:
        category = "batchnorm_small_feature_map"
        hints = [
            "Increase `imgsz` so the deepest feature maps do not collapse to 1x1 during training.",
            "Keep `batch` above 1 when possible, or reduce stride pressure by using a larger image size.",
        ]
    elif "CUDA out of memory" in text or "CUDA error" in text or "not compiled with CUDA" in text:
        category = "cuda_runtime_error"
        hints = [
            "Retry with a smaller `batch` or `imgsz` on CUDA, or let the agent fall back to `cpu` when the device was auto-selected.",
            "Check that the selected CUDA runtime matches the current PyTorch build.",
        ]
    elif "not implemented for 'MPS'" in text or "MPS backend out of memory" in text or "MPS" in text and "not supported" in text:
        category = "mps_runtime_error"
        hints = [
            "Retry with a smaller `batch` or `imgsz` while keeping `device=mps`.",
            "If the operator is unsupported on MPS, override with `runtime.device=cpu` for confirmation.",
        ]
    elif "No module named" in text:
        category = "missing_dependency"
        missing_module = detect_missing_module(text)
        if missing_module:
            info["missing_module"] = missing_module
            hints = [
                f"Install the missing dependency in the current environment, for example `python -m pip install {missing_module}`.",
                "If the import should resolve from this repo, refresh the editable install with `python -m pip install -e .`.",
            ]
        else:
            hints = [
                "Install the missing dependency inside the current Python environment before retrying.",
            ]
    elif "Dataset" in text and "not found" in text:
        category = "dataset_not_found"
        hints = [
            "Verify that the dataset YAML resolves from the current workspace and that auto-download is allowed.",
        ]

    info["category"] = category
    if hints:
        info["hints"] = hints
    return info


def tail_text(text: str, max_lines: int = 20) -> str:
    lines = strip_ansi(text).splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def cli_attempt_record(cli_result: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "cmd": cli_result["cmd"],
        "cwd": cli_result["cwd"],
        "returncode": cli_result["returncode"],
        "device": detect_cli_device(cli_result),
    }
    if cli_result["returncode"] != 0:
        record["error"] = classify_cli_failure(cli_result)
        if cli_result.get("stdout"):
            record["stdout_tail"] = tail_text(cli_result["stdout"])
        if cli_result.get("stderr"):
            record["stderr_tail"] = tail_text(cli_result["stderr"])
    return json_safe(record)


def replace_cli_device(values: dict[str, Any], device: str) -> dict[str, Any]:
    updated = dict(values)
    updated["device"] = device
    return updated


def should_retry_with_cpu(
    request: dict[str, Any],
    cli_result: dict[str, Any],
    *,
    selected_device: str | None,
    selection_source: str | None,
) -> bool:
    if cli_result["returncode"] == 0:
        return False
    if selection_source != "auto" or selected_device in (None, "cpu"):
        return False
    if not request.get("runtime", {}).get("allow_device_fallback", True):
        return False
    error = classify_cli_failure(cli_result)
    return error.get("category") in {"mps_runtime_error", "cuda_runtime_error"}


def run_cli_with_recovery(
    request: dict[str, Any],
    mode: str,
    values: dict[str, Any],
    *,
    failure_summary: str,
    selected_device: str | None,
    selection_source: str | None,
) -> dict[str, Any]:
    cli_result = run_cli(cli_args_from_values(mode, values))
    attempts = [cli_attempt_record(cli_result)]
    recovery: dict[str, Any] | None = None
    final_values = dict(values)
    final_device = detect_cli_device(cli_result) or selected_device

    if should_retry_with_cpu(
        request,
        cli_result,
        selected_device=selected_device,
        selection_source=selection_source,
    ):
        first_error = classify_cli_failure(cli_result)
        recovery = {
            "attempted": True,
            "strategy": "device_fallback_to_cpu",
            "from_device": final_device,
            "to_device": "cpu",
            "trigger": first_error,
        }
        final_values = replace_cli_device(values, "cpu")
        cli_result = run_cli(cli_args_from_values(mode, final_values))
        attempts.append(cli_attempt_record(cli_result))
        final_device = "cpu"
        recovery["recovered"] = cli_result["returncode"] == 0
        recovery["status"] = "recovered" if recovery["recovered"] else "fallback_failed"

    failed = ensure_cli_success(
        request,
        cli_result,
        failure_summary,
        attempts=attempts,
        recovery=recovery,
    )
    return {
        "cli_result": cli_result,
        "values": final_values,
        "device": final_device,
        "attempts": attempts,
        "recovery": recovery,
        "failed": failed,
    }


def ensure_cli_success(
    request: dict[str, Any],
    cli_result: dict[str, Any],
    summary: str,
    *,
    attempts: list[dict[str, Any]] | None = None,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if cli_result["returncode"] == 0:
        return None
    payload = response(
        request["skill"],
        "failed",
        summary,
        logs=cli_logs(cli_result),
        error=classify_cli_failure(cli_result),
    )
    if attempts:
        payload["attempts"] = attempts
    if recovery:
        payload["recovery"] = recovery
    return payload


def build_cli_key_values(
    request: dict[str, Any],
    *,
    skip_inputs: set[str] | None = None,
    skip_params: set[str] | None = None,
    inject_save_dir: bool = False,
) -> dict[str, Any]:
    skip_inputs = skip_inputs or set()
    skip_params = skip_params or set()
    values: dict[str, Any] = {}
    for key, value in request["inputs"].items():
        if key in skip_inputs or value is None:
            continue
        values[key] = value
    for key, value in request["params"].items():
        if key in skip_params or value is None:
            continue
        values[key] = value
    if inject_save_dir:
        values = inject_cli_artifact_location(request, values)
    return values


def cli_args_from_values(mode: str, values: dict[str, Any]) -> list[str]:
    args = [mode]
    for key, value in values.items():
        args.append(kv_arg(key, value))
    return args


def read_results_csv_metrics(save_dir: Path | None) -> dict[str, Any]:
    if not save_dir:
        return {}
    results_csv = Path(save_dir) / "results.csv"
    if not results_csv.exists():
        return {}
    try:
        with results_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return {}
    if not rows:
        return {}
    return {key: coerce_scalar(value) for key, value in rows[-1].items()}


def parse_cli_speed(stdout: str) -> dict[str, float]:
    speed: dict[str, float] = {}
    pattern = re.compile(
        r"^Speed:\s+([\d.]+)ms preprocess,\s+([\d.]+)ms inference(?:,\s+([\d.]+)ms loss)?,\s+([\d.]+)ms postprocess"
    )
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        match = pattern.match(line)
        if not match:
            continue
        speed = {
            "preprocess": float(match.group(1)),
            "inference": float(match.group(2)),
            "postprocess": float(match.group(4)),
        }
        if match.group(3) is not None:
            speed["loss"] = float(match.group(3))
    return speed


def parse_detection_cli_metrics(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pattern = re.compile(r"^all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
    for raw_line in reversed(stdout.splitlines()):
        line = raw_line.strip()
        match = pattern.match(line)
        if not match:
            continue
        images = int(match.group(1))
        instances = int(match.group(2))
        precision = float(match.group(3))
        recall = float(match.group(4))
        map50 = float(match.group(5))
        map50_95 = float(match.group(6))
        raw_metrics = {
            "metrics/precision(B)": precision,
            "metrics/recall(B)": recall,
            "metrics/mAP50(B)": map50,
            "metrics/mAP50-95(B)": map50_95,
        }
        evaluation = {
            "images": images,
            "instances": instances,
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
        }
        return raw_metrics, evaluation
    return {}, {}


def build_evaluation_summary(metrics: dict[str, Any], stdout: str = "") -> dict[str, Any]:
    _, parsed = parse_detection_cli_metrics(stdout)
    evaluation = dict(parsed)
    mapping = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
        "train/box_loss": "train_box_loss",
        "train/cls_loss": "train_cls_loss",
        "train/dfl_loss": "train_dfl_loss",
        "train/moe_loss": "train_moe_loss",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
        "val/dfl_loss": "val_dfl_loss",
        "val/moe_loss": "val_moe_loss",
        "epoch": "epoch",
        "time": "time_sec",
    }
    for source_key, target_key in mapping.items():
        if source_key in metrics:
            evaluation[target_key] = metrics[source_key]
    speed = parse_cli_speed(stdout)
    if speed:
        evaluation["speed_ms"] = speed
    return evaluation


def parse_predict_cli_output(stdout: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    results: list[dict[str, Any]] = []
    speed: dict[str, float] = parse_cli_speed(stdout)
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        match = re.match(r"^image\s+\d+/\d+\s+(.*?):\s+(.*)$", line)
        if match:
            item: dict[str, Any] = {"path": match.group(1), "raw": match.group(2)}
            if "no detections" in match.group(2).lower():
                item["boxes"] = 0
            results.append(item)
    return results, speed


def capture_output(func, *args, **kwargs) -> tuple[Any, str, str]:
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        result = func(*args, **kwargs)
    return result, stdout_buffer.getvalue(), stderr_buffer.getvalue()


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def ensure_manifest_dir(request: dict[str, Any]) -> Path:
    project = request.get("artifacts", {}).get("project")
    name = request.get("artifacts", {}).get("name")
    base = (REPO_ROOT / project).resolve() if project else DEFAULT_MANIFEST_DIR / request["request_id"]
    target = base / name if name else base
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_manifest(request: dict[str, Any], payload: dict[str, Any]) -> Path:
    manifest_dir = ensure_manifest_dir(request)
    manifest_path = manifest_dir / "skill_manifest.json"
    manifest = {
        "skill": request.get("skill"),
        "request_id": request.get("request_id"),
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "artifacts": payload.get("artifacts", []),
        "metrics": payload.get("metrics", {}),
        "evaluation": payload.get("evaluation", {}),
        "environment": payload.get("environment", {}),
        "auto_completed": payload.get("auto_completed", {}),
        "attempts": payload.get("attempts", []),
        "recovery": payload.get("recovery", {}),
        "multimodal": payload.get("multimodal", {}),
        "recommendations": payload.get("recommendations", []),
        "job": payload.get("job", {}),
        "dry_run": payload.get("dry_run", False),
    }
    manifest_path.write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def finalize_payload(request: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("request_id", request.get("request_id"))
    if "manifest" not in payload:
        payload["manifest"] = str(write_manifest(request, payload))
    return json_safe(payload)


def best_checkpoint(payload: dict[str, Any]) -> str | None:
    for artifact in payload.get("artifacts", []):
        if artifact.get("kind") == "checkpoint" and artifact.get("label") == "best":
            return artifact.get("path")
    for artifact in payload.get("artifacts", []):
        if artifact.get("kind") == "checkpoint":
            return artifact.get("path")
    return None


def metrics_payload(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    if hasattr(metrics, "results_dict"):
        return json_safe(metrics.results_dict)
    if isinstance(metrics, dict):
        return json_safe(metrics)
    return {"value": json_safe(metrics)}


def summarize_results(results: Any, max_items: int = 10) -> list[dict[str, Any]]:
    summary = []
    iterable = list(results)
    for result in iterable[:max_items]:
        item: dict[str, Any] = {
            "path": str(getattr(result, "path", "")),
            "speed": json_safe(getattr(result, "speed", {})),
        }
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        probs = getattr(result, "probs", None)
        obb = getattr(result, "obb", None)
        if boxes is not None:
            try:
                item["boxes"] = len(boxes)
            except Exception:
                item["boxes"] = 0
        if masks is not None:
            try:
                item["masks"] = len(masks)
            except Exception:
                item["masks"] = 0
        if obb is not None:
            try:
                item["obb"] = len(obb)
            except Exception:
                item["obb"] = 0
        if probs is not None:
            item["classification"] = {
                "top1": json_safe(getattr(probs, "top1", None)),
                "top1conf": json_safe(getattr(probs, "top1conf", None)),
            }
        summary.append(item)
    return summary


def summarize_results_for_reasoning(results: Any, max_items: int = 5, max_boxes: int = 20) -> list[dict[str, Any]]:
    """Return compact, structured prediction evidence for multimodal reasoning."""
    summary = []
    for result in list(results)[:max_items]:
        item: dict[str, Any] = {
            "path": str(getattr(result, "path", "")),
            "speed": json_safe(getattr(result, "speed", {})),
            "detections": [],
        }
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            try:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                cls = boxes.cls.detach().cpu().tolist()
                conf = boxes.conf.detach().cpu().tolist()
                for idx, coords in enumerate(xyxy[:max_boxes]):
                    class_id = int(cls[idx]) if idx < len(cls) else None
                    item["detections"].append(
                        {
                            "class_id": class_id,
                            "label": names.get(class_id, str(class_id)) if class_id is not None else None,
                            "confidence": round(float(conf[idx]), 4) if idx < len(conf) else None,
                            "xyxy": [round(float(v), 2) for v in coords],
                        }
                    )
            except Exception:
                try:
                    item["boxes"] = len(boxes)
                except Exception:
                    item["boxes"] = 0
        summary.append(item)
    return summary


def image_source_for_openai(source: Any, results_summary: list[dict[str, Any]]) -> str | None:
    candidates: list[str] = []
    if source is not None:
        candidates.append(str(source))
    if results_summary and results_summary[0].get("path"):
        candidate = str(results_summary[0]["path"])
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.startswith(("http://", "https://", "data:image/")):
            return candidate
        path = resolved_path(candidate)
        if path.exists() and path.is_file():
            return str(path)
    return None


def encode_image_reference_for_openai(image_ref: str, max_bytes: int = 20_000_000) -> dict[str, Any]:
    if image_ref.startswith(("http://", "https://", "data:image/")):
        return {"image_url": image_ref, "kind": "url" if image_ref.startswith(("http://", "https://")) else "data_url"}
    path = resolved_path(image_ref)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image source for VLM is not a file or URL: {image_ref}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Image source is too large for inline VLM upload: {path} ({size} bytes)")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"image_url": f"data:{mime_type};base64,{data}", "kind": "local_file", "path": str(path.resolve())}


def openai_config(params: dict[str, Any]) -> dict[str, Any]:
    provider = params.get("vlm_provider") or params.get("provider") or "openai"
    base_url = str(params.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    return {
        "provider": provider,
        "api_key_env": "OPENAI_API_KEY",
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "base_url": base_url,
        "api_mode": params.get("openai_api_mode") or os.environ.get("OPENAI_API_MODE") or "auto",
        "vlm_model": params.get("vlm_model") or os.environ.get("OPENAI_VLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini",
        "llm_model": params.get("llm_model") or os.environ.get("OPENAI_LLM_MODEL") or os.environ.get("OPENAI_MODEL"),
    }


def build_thinking_with_image_prompt(
    user_prompt: str,
    detections: list[dict[str, Any]],
    *,
    method: str = "thinking-with-image",
    thinking_with_image: bool = True,
    structured_output: bool = True,
) -> str:
    detection_text = json.dumps(json_safe(detections), ensure_ascii=False, indent=2)
    image_instruction = (
        "Privately inspect the image, compare it with the YOLO detection summary, and resolve disagreements."
        if thinking_with_image
        else "Use the user prompt and YOLO detection summary as the evidence surface, and do not assume image access."
    )
    output_instruction = (
        "Return exactly one JSON object without Markdown fences. Use these keys: answer, visual_evidence, "
        "yolo_cross_check, uncertainty, recommended_next_actions. In yolo_cross_check, include arrays named "
        "confirmed, false_positives, possible_misses, duplicate_or_fragmented, and notes when applicable."
        if structured_output
        else "Return these sections: answer, visual_evidence, yolo_cross_check, uncertainty, recommended_next_actions."
    )
    return (
        "You are helping a YOLO-Master agent perform multimodal visual inference.\n"
        f"Method: {method}. {image_instruction} Do not reveal hidden chain-of-thought. "
        "Return a concise, evidence-based answer.\n\n"
        f"User task:\n{user_prompt}\n\n"
        f"YOLO detection summary:\n{detection_text}\n\n{output_instruction}"
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    def balanced_fragment(source: str) -> str | None:
        start = None
        depth = 0
        in_string = False
        escape = False
        opener = None
        for idx, ch in enumerate(source):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                if start is None:
                    start = idx
                    opener = ch
                    depth = 1
                    continue
                depth += 1
            elif ch in '}]':
                if start is None:
                    continue
                depth -= 1
                if depth == 0 and opener is not None:
                    return source[start : idx + 1]
        return None

    fragment = balanced_fragment(cleaned)
    if fragment is not None:
        try:
            value = json.loads(fragment)
            return value if isinstance(value, dict) else None
        except Exception:
            pass

    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{]", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def attach_multimodal_verdict(result: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    if result.get("status") == "ok" and "verdict" not in result:
        verdict = extract_json_object(str(result.get("text") or ""))
        expected_keys = {"answer", "visual_evidence", "yolo_cross_check", "uncertainty", "recommended_next_actions"}
        if verdict is not None and expected_keys.intersection(verdict):
            result["verdict"] = json_safe(verdict)
            result["verdict_parse_status"] = "parsed"
        else:
            result["verdict_parse_status"] = "unparsed"
    return result


def extract_openai_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for output in payload.get("output", []) or []:
        for content in output.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def extract_openai_chat_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for choice in payload.get("choices", []) or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
    return "\n".join(chunks).strip()


def classify_openai_http_status(detail: str) -> str:
    text = detail.lower()
    blocked_markers = (
        "access denied",
        "arrearage",
        "insufficient_quota",
        "quota",
        "billing",
        "permission",
        "unauthorized",
        "forbidden",
    )
    return "blocked" if any(marker in text for marker in blocked_markers) else "failed"


def call_openai_responses(
    *,
    model: str,
    user_text: str,
    developer_text: str | None = None,
    image_url: str | None = None,
    image_detail: str = "auto",
    base_url: str | None = None,
    max_output_tokens: int = 800,
    temperature: float | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "blocked",
            "provider": "openai",
            "summary": "OPENAI_API_KEY is not set; multimodal reasoning was skipped.",
            "api_key_env": "OPENAI_API_KEY",
        }

    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    if image_url:
        content.append({"type": "input_image", "image_url": image_url, "detail": image_detail})
    input_items = []
    if developer_text:
        input_items.append({"role": "developer", "content": [{"type": "input_text", "text": developer_text}]})
    input_items.append({"role": "user", "content": content})

    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature

    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response_handle:
            payload = json.loads(response_handle.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        status = classify_openai_http_status(detail)
        return {
            "status": status,
            "provider": "openai",
            "summary": f"OpenAI Responses API returned HTTP {exc.code}",
            "error": {"type": "HTTPError", "code": exc.code, "body": detail},
        }
    except Exception as exc:
        return {
            "status": "failed",
            "provider": "openai",
            "summary": "OpenAI Responses API request failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    return {
        "status": "ok",
        "provider": "openai",
        "api_mode": "responses",
        "model": model,
        "text": extract_openai_text(payload),
        "response_id": payload.get("id"),
        "usage": json_safe(payload.get("usage", {})),
    }


def call_openai_chat_completions(
    *,
    model: str,
    user_text: str,
    developer_text: str | None = None,
    image_url: str | None = None,
    base_url: str | None = None,
    max_output_tokens: int = 800,
    temperature: float | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "blocked",
            "provider": "openai",
            "api_mode": "chat.completions",
            "summary": "OPENAI_API_KEY is not set; multimodal reasoning was skipped.",
            "api_key_env": "OPENAI_API_KEY",
        }

    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = []
    if developer_text:
        messages.append({"role": "system", "content": developer_text})
    messages.append({"role": "user", "content": content})

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response_handle:
            payload = json.loads(response_handle.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        status = classify_openai_http_status(detail)
        return {
            "status": status,
            "provider": "openai",
            "api_mode": "chat.completions",
            "summary": f"OpenAI-compatible Chat Completions API returned HTTP {exc.code}",
            "error": {"type": "HTTPError", "code": exc.code, "body": detail},
        }
    except Exception as exc:
        return {
            "status": "failed",
            "provider": "openai",
            "api_mode": "chat.completions",
            "summary": "OpenAI-compatible Chat Completions API request failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    return {
        "status": "ok",
        "provider": "openai",
        "api_mode": "chat.completions",
        "model": model,
        "text": extract_openai_chat_text(payload),
        "response_id": payload.get("id"),
        "usage": json_safe(payload.get("usage", {})),
    }


def call_openai_compatible(
    *,
    model: str,
    user_text: str,
    developer_text: str | None = None,
    image_url: str | None = None,
    image_detail: str = "auto",
    base_url: str | None = None,
    api_mode: str = "auto",
    max_output_tokens: int = 800,
    temperature: float | None = None,
) -> dict[str, Any]:
    normalized_mode = api_mode.replace("_", ".").lower()
    if normalized_mode in {"chat", "chat.completion", "chat.completions"}:
        return call_openai_chat_completions(
            model=model,
            user_text=user_text,
            developer_text=developer_text,
            image_url=image_url,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    if normalized_mode == "responses":
        return call_openai_responses(
            model=model,
            user_text=user_text,
            developer_text=developer_text,
            image_url=image_url,
            image_detail=image_detail,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    responses_result = call_openai_responses(
        model=model,
        user_text=user_text,
        developer_text=developer_text,
        image_url=image_url,
        image_detail=image_detail,
        base_url=base_url,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    if responses_result.get("status") in {"ok", "blocked"}:
        return responses_result
    chat_result = call_openai_chat_completions(
        model=model,
        user_text=user_text,
        developer_text=developer_text,
        image_url=image_url,
        base_url=base_url,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    if chat_result.get("status") == "ok":
        chat_result["fallback_from"] = responses_result
    return chat_result


def multimodal_overall_status(vlm_result: dict[str, Any], llm_result: dict[str, Any] | None) -> str:
    vlm_status = vlm_result.get("status")
    if vlm_status == "blocked" or (llm_result and llm_result.get("status") == "blocked"):
        return "blocked"
    if vlm_status != "ok":
        return "partial"
    if llm_result and llm_result.get("status") == "failed":
        return "partial"
    return "ok"


def response(skill: str, status: str, summary: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"skill": skill, "status": status, "summary": summary}
    payload.update(kwargs)
    return json_safe(payload)


def plan_response(
    request: dict[str, Any],
    summary: str,
    executor: str,
    target: str,
    params: dict[str, Any] | None = None,
    next_actions: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = response(
        request["skill"],
        "ok",
        summary,
        dry_run=True,
        plan={
            "executor": executor,
            "target": target,
            "inputs": json_safe(request.get("inputs", {})),
            "params": json_safe(params if params is not None else request.get("params", {})),
        },
        next_actions=next_actions or [],
    )
    if extra:
        payload.update(json_safe(extra))
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def build_model(request: dict[str, Any]) -> Any:
    inputs = request["inputs"]
    model_ref = inputs.get("model")
    if not model_ref:
        raise ValueError("`inputs.model` is required.")
    YOLO = get_ultralytics_core()["YOLO"]
    return YOLO(model_ref, task=inputs.get("task"))


def run_system(request: dict[str, Any]) -> dict[str, Any]:
    action = request.get("action") or request["params"].get("action") or "help"
    params = request["params"]

    if is_dry_run(request):
        if action == "install":
            return plan_response(
                request,
                "system install dry run prepared",
                "bootstrap",
                "pip install -e .",
                params={"cmd": cli_install_command()},
            )
        if action == "doctor":
            selected_device = resolve_default_device(request, params)
            environment = collect_environment_report(request, selected_device=selected_device)
            recommendations = doctor_recommendations(environment)
            return plan_response(
                request,
                "system doctor dry run prepared",
                "module",
                "yolo.system::doctor",
                params=params,
                extra={"environment": environment, "recommendations": recommendations},
            )
        cli_map = {
            "help": ["help"],
            "version": ["version"],
            "checks": ["checks"],
            "settings.get": ["settings"],
            "settings.update": ["settings", *[kv_arg(k, v) for k, v in (params.get("updates") or {k: v for k, v in params.items() if k != "action"}).items()]],
            "settings.reset": ["settings", "reset"],
            "cfg.get": ["cfg"],
            "cfg.copy": ["copy-cfg"],
        }
        if action in cli_map:
            return cli_plan(request, cli_map[action])
        return plan_response(request, "system dry run prepared", "module", f"yolo.system::{action}", params=params)

    if action == "install":
        install = install_ultralytics_cli()
        return response(
            request["skill"],
            "ok" if install["returncode"] == 0 else "failed",
            "ultralytics CLI installed" if install["returncode"] == 0 else "ultralytics CLI install failed",
            data={"install": install, "yolo": find_yolo_cli()},
        )

    if action == "doctor":
        force_install = bool(params.get("force_install", False))
        ensure_cli = bool(params.get("ensure_cli", True))
        if ensure_cli:
            _, install = ensure_yolo_cli(force_install=force_install)
        else:
            cli_path = find_yolo_cli()
            install = {"status": "available" if cli_path else "missing", "path": cli_path}
        selected_device = resolve_default_device(request, params)
        environment = collect_environment_report(request, selected_device=selected_device, cli_info=install)
        recommendations = doctor_recommendations(environment)
        return response(
            request["skill"],
            "ok",
            "environment doctor collected",
            data={"environment": environment, "recommendations": recommendations},
            environment=environment,
            recommendations=recommendations,
        )

    if action in {"help", "version", "checks", "settings.get", "settings.update", "settings.reset", "cfg.get", "cfg.copy"}:
        cli_args = {
            "help": ["help"],
            "version": ["version"],
            "checks": ["checks"],
            "settings.get": ["settings"],
            "settings.update": ["settings", *[kv_arg(k, v) for k, v in (params.get("updates") or {k: v for k, v in params.items() if k != "action"}).items()]],
            "settings.reset": ["settings", "reset"],
            "cfg.get": ["cfg"],
            "cfg.copy": ["copy-cfg"],
        }[action]
        cwd = ensure_manifest_dir(request) if action == "cfg.copy" else None
        cli_result = run_cli(cli_args, cwd=cwd)
        failed = ensure_cli_success(request, cli_result, f"system action `{action}` failed")
        if failed:
            return failed
        if action == "help":
            return response(
                request["skill"],
                "ok",
                "available system actions",
                actions=["install", "doctor", "help", "version", "checks", "settings.get", "settings.update", "settings.reset", "cfg.get", "cfg.copy"],
                logs=cli_logs(cli_result),
            )
        if action == "version":
            match = re.search(r"\b\d+\.\d+\.\d+\b", f"{cli_result['stdout']}\n{cli_result['stderr']}")
            version = match.group(0) if match else read_repo_version()
            return response(request["skill"], "ok", "version collected", data={"version": version}, logs=cli_logs(cli_result))
        if action == "checks":
            return response(request["skill"], "ok", "system checks collected", logs=cli_logs(cli_result))
        if action == "settings.get":
            core = get_ultralytics_core()
            return response(
                request["skill"],
                "ok",
                "settings collected",
                data={"settings": json_safe(dict(core["SETTINGS"]))},
                logs=cli_logs(cli_result),
            )
        if action == "settings.update":
            core = get_ultralytics_core()
            updates = params.get("updates") or {k: v for k, v in params.items() if k != "action"}
            return response(
                request["skill"],
                "ok",
                "settings updated",
                data={"settings": json_safe(dict(core["SETTINGS"])), "updated": json_safe(updates)},
                logs=cli_logs(cli_result),
            )
        if action == "settings.reset":
            core = get_ultralytics_core()
            return response(request["skill"], "ok", "settings reset", data={"settings": json_safe(dict(core["SETTINGS"]))}, logs=cli_logs(cli_result))
        if action == "cfg.get":
            return response(request["skill"], "ok", "default cfg loaded", data={"cfg": json_safe(read_default_cfg())}, logs=cli_logs(cli_result))
        if action == "cfg.copy":
            new_file = ensure_manifest_dir(request) / DEFAULT_CFG_FILE.name.replace(".yaml", "_copy.yaml")
            return response(
                request["skill"],
                "ok",
                "default cfg copied",
                artifacts=[{"kind": "config", "path": str(new_file.resolve())}],
                logs=cli_logs(cli_result),
            )
    raise ValueError(f"Unsupported yolo.system action: {action}")


def run_model_inspect(request: dict[str, Any]) -> dict[str, Any]:
    actions = request["params"].get("actions") or ["info", "names", "device", "task_map"]
    if is_dry_run(request):
        return plan_response(request, "inspect dry run prepared", "python_api", "YOLO(...).inspect", params={"actions": actions})

    model = build_model(request)
    data: dict[str, Any] = {"task": model.task, "model_name": json_safe(getattr(model, "model_name", None))}
    for action in actions:
        if action == "info":
            data["info"] = json_safe(model.info(verbose=False))
        elif action == "names":
            data["names"] = json_safe(model.names)
        elif action == "device":
            data["device"] = str(model.device)
        elif action == "task_map":
            data["task_map"] = {k: list(v.keys()) for k, v in model.task_map.items()}
        elif action == "fuse":
            model.fuse()
            data["fused"] = True
        elif action == "reset_weights":
            model.reset_weights()
            data["reset_weights"] = True
        else:
            raise ValueError(f"Unsupported inspect action: {action}")
    try:
        model._check_is_pytorch_model()
        data["supports_pytorch_only"] = True
    except Exception:
        data["supports_pytorch_only"] = False
    payload = response(request["skill"], "ok", "model inspected", data=data)
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_train_like(request: dict[str, Any], skill_name: str) -> dict[str, Any]:
    params = dict(request["params"])
    if request["inputs"].get("data") and "data" not in params:
        params["data"] = request["inputs"]["data"]
    device_selection = resolve_device_selection(request, params)
    params, chosen_device, auto_completed = apply_runtime_defaults(request, params, purpose="train")
    effective_request = deepcopy(request)
    effective_request["params"] = params
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
            return cli_plan(
                request,
                cli_args_from_values("train", values),
                extra={
                    "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                    "auto_completed": auto_completed,
                },
            )
        return plan_response(
            request,
            "training dry run prepared",
            "python_api",
            "YOLO(...).train",
            params=params,
            next_actions=["yolo.val", "yolo.export"],
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
            },
        )

    if prefer_cli(request):
        values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
        cli_execution = run_cli_with_recovery(
            request,
            "train",
            values,
            failure_summary="training failed",
            selected_device=chosen_device,
            selection_source=device_selection["source"],
        )
        failed = cli_execution["failed"]
        if failed:
            return failed
        cli_result = cli_execution["cli_result"]
        values = cli_execution["values"]
        final_device = cli_execution["device"]
        recovery = cli_execution["recovery"]
        save_dir = cli_save_dir(request, values)
        artifacts = []
        metrics = read_results_csv_metrics(save_dir)
        parsed_metrics, _ = parse_detection_cli_metrics(cli_result["stdout"])
        for key, value in parsed_metrics.items():
            metrics.setdefault(key, value)
        evaluation = build_evaluation_summary(metrics, cli_result["stdout"])
        environment = collect_environment_report(
            effective_request,
            selected_device=final_device,
            requested_device=chosen_device,
            selection_source="recovery" if recovery and recovery.get("recovered") else device_selection["source"],
            cli_info=cli_result["install"],
        )
        if save_dir and save_dir.exists():
            best = save_dir / "weights" / "best.pt"
            last = save_dir / "weights" / "last.pt"
            if best.exists():
                artifacts.append({"kind": "checkpoint", "label": "best", "path": str(best.resolve())})
            if last.exists():
                artifacts.append({"kind": "checkpoint", "label": "last", "path": str(last.resolve())})
            if (save_dir / "results.csv").exists():
                artifacts.append({"kind": "csv", "path": str((save_dir / "results.csv").resolve())})
            if (save_dir / "args.yaml").exists():
                artifacts.append({"kind": "config", "path": str((save_dir / "args.yaml").resolve())})
        return response(
            skill_name,
            "ok",
            "training finished after automatic cpu fallback" if recovery and recovery.get("recovered") else "training finished",
            job={
                "mode": "sync",
                "save_dir": json_safe(save_dir),
                "resume_supported": True,
                "executor": "cli",
                "device": final_device,
            },
            metrics=metrics,
            evaluation=evaluation,
            environment=environment,
            auto_completed=auto_completed,
            artifacts=artifacts,
            logs=cli_logs(cli_result),
            attempts=cli_execution["attempts"] if recovery else [],
            recovery=recovery or {},
            next_actions=["yolo.val", "yolo.export"],
        )

    model = build_model(request)
    metrics = model.train(**params)
    environment = collect_environment_report(effective_request, selected_device=chosen_device)
    artifacts = []
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        save_dir = Path(save_dir)
        best = getattr(trainer, "best", None)
        last = getattr(trainer, "last", None)
        if best and Path(best).exists():
            artifacts.append({"kind": "checkpoint", "label": "best", "path": str(Path(best).resolve())})
        if last and Path(last).exists():
            artifacts.append({"kind": "checkpoint", "label": "last", "path": str(Path(last).resolve())})
        if (save_dir / "results.csv").exists():
            artifacts.append({"kind": "csv", "path": str((save_dir / "results.csv").resolve())})
        if (save_dir / "args.yaml").exists():
            artifacts.append({"kind": "config", "path": str((save_dir / "args.yaml").resolve())})
    payload = response(
        skill_name,
        "ok",
        "training finished",
        job={"mode": "sync", "save_dir": json_safe(save_dir), "resume_supported": True, "device": chosen_device},
        metrics=metrics_payload(metrics or getattr(model, "metrics", None)),
        environment=environment,
        auto_completed=auto_completed,
        artifacts=artifacts,
        next_actions=["yolo.val", "yolo.export"],
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_val(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request["params"])
    if request["inputs"].get("data") and "data" not in params:
        params["data"] = request["inputs"]["data"]
    device_selection = resolve_device_selection(request, params)
    params, chosen_device, auto_completed = apply_runtime_defaults(request, params, purpose="val")
    effective_request = deepcopy(request)
    effective_request["params"] = params
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
            return cli_plan(
                request,
                cli_args_from_values("val", values),
                extra={
                    "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                    "auto_completed": auto_completed,
                },
            )
        return plan_response(
            request,
            "validation dry run prepared",
            "python_api",
            "YOLO(...).val",
            params=params,
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
            },
        )

    if prefer_cli(request):
        values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
        cli_execution = run_cli_with_recovery(
            request,
            "val",
            values,
            failure_summary="validation failed",
            selected_device=chosen_device,
            selection_source=device_selection["source"],
        )
        failed = cli_execution["failed"]
        if failed:
            return failed
        cli_result = cli_execution["cli_result"]
        values = cli_execution["values"]
        final_device = cli_execution["device"]
        recovery = cli_execution["recovery"]
        save_dir = cli_save_dir(request, values)
        artifacts = []
        metrics, evaluation = parse_detection_cli_metrics(cli_result["stdout"])
        speed = parse_cli_speed(cli_result["stdout"])
        evaluation = build_evaluation_summary(metrics, cli_result["stdout"]) if metrics else ({"speed_ms": speed} if speed else {})
        environment = collect_environment_report(
            effective_request,
            selected_device=final_device,
            requested_device=chosen_device,
            selection_source="recovery" if recovery and recovery.get("recovered") else device_selection["source"],
            cli_info=cli_result["install"],
        )
        if save_dir and (save_dir / "predictions.json").exists():
            artifacts.append({"kind": "json", "path": str((save_dir / "predictions.json").resolve())})
        return response(
            request["skill"],
            "ok",
            "validation finished after automatic cpu fallback" if recovery and recovery.get("recovered") else "validation finished",
            metrics=metrics,
            evaluation=evaluation,
            environment=environment,
            auto_completed=auto_completed,
            job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "cli", "device": final_device},
            artifacts=artifacts,
            logs=cli_logs(cli_result),
            attempts=cli_execution["attempts"] if recovery else [],
            recovery=recovery or {},
        )

    model = build_model(request)
    metrics = model.val(**params)
    environment = collect_environment_report(effective_request, selected_device=chosen_device)
    artifacts = []
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir:
        save_dir = Path(save_dir)
        if (save_dir / "predictions.json").exists():
            artifacts.append({"kind": "json", "path": str((save_dir / "predictions.json").resolve())})
    payload = response(
        request["skill"],
        "ok",
        "validation finished",
        metrics=metrics_payload(metrics),
        environment=environment,
        auto_completed=auto_completed,
        job={"mode": "sync", "save_dir": json_safe(save_dir), "device": chosen_device},
        artifacts=artifacts,
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_predict_like(request: dict[str, Any], mode: str) -> dict[str, Any]:
    params = dict(request["params"])
    source = request["inputs"].get("source") or params.pop("source", None)
    if source is None:
        raise ValueError("`inputs.source` is required for predict/track.")
    max_items = int(params.pop("max_items", 10))
    device_selection = resolve_device_selection(request, params)
    params, chosen_device, auto_completed = apply_runtime_defaults(request, params, purpose=mode)
    effective_request = deepcopy(request)
    effective_request["params"] = params
    effective_request["inputs"]["source"] = source
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params={"max_items"}, inject_save_dir=True)
            return cli_plan(
                effective_request,
                cli_args_from_values(mode, values),
                extra={
                    "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                    "auto_completed": auto_completed,
                },
            )
        target = "YOLO(...).predict" if mode == "predict" else "YOLO(...).track"
        plan_params = {"source": source, **params}
        return plan_response(
            effective_request,
            f"{mode} dry run prepared",
            "python_api",
            target,
            params=plan_params,
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
            },
        )

    if prefer_cli(request):
        values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params={"max_items"}, inject_save_dir=True)
        cli_execution = run_cli_with_recovery(
            request,
            mode,
            values,
            failure_summary=f"{mode} failed",
            selected_device=chosen_device,
            selection_source=device_selection["source"],
        )
        failed = cli_execution["failed"]
        if failed:
            return failed
        cli_result = cli_execution["cli_result"]
        values = cli_execution["values"]
        final_device = cli_execution["device"]
        recovery = cli_execution["recovery"]
        save_dir = cli_save_dir(request, values)
        results, speed = parse_predict_cli_output(cli_result["stdout"])
        environment = collect_environment_report(
            effective_request,
            selected_device=final_device,
            requested_device=chosen_device,
            selection_source="recovery" if recovery and recovery.get("recovered") else device_selection["source"],
            cli_info=cli_result["install"],
        )
        payload = response(
            request["skill"],
            "ok",
            f"{mode} finished after automatic cpu fallback" if recovery and recovery.get("recovered") else f"{mode} finished",
            job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "cli", "device": final_device},
            results=results[:max_items],
            environment=environment,
            auto_completed=auto_completed,
            logs=cli_logs(cli_result),
            attempts=cli_execution["attempts"] if recovery else [],
            recovery=recovery or {},
        )
        if speed and payload["results"]:
            payload["results"][0]["speed"] = speed
        if save_dir and save_dir.exists():
            payload["artifacts"] = [{"kind": "directory", "path": str(save_dir.resolve())}]
        return payload

    model = build_model(request)
    if mode == "predict":
        results = model.predict(source=source, **params)
    else:
        results = model.track(source=source, **params)
    save_dir = getattr(model.predictor, "save_dir", None)
    payload = response(
        request["skill"],
        "ok",
        f"{mode} finished",
        job={"mode": "sync", "save_dir": json_safe(save_dir), "device": chosen_device},
        environment=collect_environment_report(effective_request, selected_device=chosen_device),
        auto_completed=auto_completed,
        results=summarize_results(results, max_items=max_items),
    )
    if save_dir:
        payload["artifacts"] = [{"kind": "directory", "path": str(Path(save_dir).resolve())}]
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def split_yolo_and_multimodal_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    yolo_params = {}
    multimodal_params = {}
    for key, value in params.items():
        if key in MULTIMODAL_PARAM_KEYS:
            multimodal_params[key] = value
        else:
            yolo_params[key] = value
    return yolo_params, multimodal_params


def split_yolo_multimodal_evaluate_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    yolo_params: dict[str, Any] = {}
    multimodal_params: dict[str, Any] = {}
    evaluate_params: dict[str, Any] = {}
    for key, value in params.items():
        if key in MULTIMODAL_PARAM_KEYS:
            multimodal_params[key] = value
        elif key in MULTIMODAL_EVALUATE_PARAM_KEYS:
            evaluate_params[key] = value
        else:
            yolo_params[key] = value
    return yolo_params, multimodal_params, evaluate_params


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def read_image_list_file(path: Path, root: Path | None = None) -> list[Path]:
    images: list[Path] = []
    base = root or path.parent
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        if is_image_file(candidate):
            images.append(candidate)
    return images


def expand_image_reference(value: Any, root: Path | None = None) -> list[Path]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        images: list[Path] = []
        for item in value:
            images.extend(expand_image_reference(item, root=root))
        return images
    text = str(value)
    if text.startswith(("http://", "https://", "data:image/")):
        return []
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = ((root or REPO_ROOT) / candidate).resolve()
    if candidate.is_dir():
        return sorted(path.resolve() for path in candidate.rglob("*") if is_image_file(path))
    if candidate.is_file():
        if candidate.suffix.lower() == ".txt":
            return read_image_list_file(candidate, root=root)
        if is_image_file(candidate):
            return [candidate.resolve()]
    return []


def normalize_dataset_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        normalized = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except Exception:
                continue
        return normalized
    if isinstance(names, list):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def load_dataset_yaml(data_ref: Any) -> tuple[Path, dict[str, Any]]:
    import yaml

    if data_ref in (None, ""):
        raise ValueError("`inputs.data` or `params.data` is required when no image `source` is provided.")
    normalized = normalize_value(data_ref)
    data_path = Path(str(normalized))
    if not data_path.is_absolute():
        data_path = resolved_path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML was not found: {data_ref}")
    loaded = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_path}")
    return data_path, loaded


def dataset_settings_dir() -> Path | None:
    try:
        settings = dict(get_ultralytics_core()["SETTINGS"])
    except Exception:
        return None
    value = settings.get("datasets_dir")
    return Path(str(value)).expanduser().resolve() if value else None


def resolve_dataset_root(data_path: Path, dataset_cfg: dict[str, Any]) -> Path:
    path_value = dataset_cfg.get("path")
    if path_value in (None, ""):
        return data_path.parent.resolve()
    candidate_path = Path(str(path_value)).expanduser()
    if candidate_path.is_absolute():
        return candidate_path.resolve()

    candidates = [
        (data_path.parent / candidate_path).resolve(),
        (REPO_ROOT / candidate_path).resolve(),
    ]
    settings_dir = dataset_settings_dir()
    if settings_dir is not None:
        candidates.append((settings_dir / candidate_path).resolve())
    candidates.append((REPO_ROOT.parent / "datasets" / candidate_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def dataset_split_spec(dataset_cfg: dict[str, Any], requested_split: str) -> tuple[str, Any]:
    for split in (requested_split, "val", "train", "test"):
        value = dataset_cfg.get(split)
        if value not in (None, ""):
            return split, value
    raise ValueError(f"Dataset YAML has no usable split for `{requested_split}`.")


def collect_dataset_images(data_ref: Any, split: str) -> tuple[list[Path], dict[str, Any], dict[int, str]]:
    data_path, dataset_cfg = load_dataset_yaml(data_ref)
    root = resolve_dataset_root(data_path, dataset_cfg)
    actual_split, spec = dataset_split_spec(dataset_cfg, split)
    images = expand_image_reference(spec, root=root)
    names = normalize_dataset_names(dataset_cfg.get("names", {}))
    dataset_info = {
        "data": str(data_path),
        "root": str(root),
        "split": actual_split,
        "requested_split": split,
        "source": json_safe(spec),
        "names_count": len(names),
    }
    return images, dataset_info, names


def dedupe_images(images: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for image in images:
        key = str(image.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(image.resolve())
    return unique


def select_image_sample(
    images: list[Path],
    *,
    limit: int | None,
    offset: int = 0,
    stride: int = 1,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[Path]:
    selected = list(images)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if offset > 0:
        selected = selected[offset:]
    if stride > 1:
        selected = selected[::stride]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def collect_multimodal_evaluation_images(
    request: dict[str, Any],
    evaluate_params: dict[str, Any],
    yolo_params: dict[str, Any],
    multimodal_params: dict[str, Any],
) -> tuple[list[Path], dict[str, Any], dict[int, str]]:
    split = str(evaluate_params.get("split", "val"))
    source_ref = request["inputs"].get("source") or yolo_params.pop("source", None) or multimodal_params.get("source")
    data_ref = request["inputs"].get("data") or evaluate_params.get("data") or yolo_params.pop("data", None)
    if source_ref not in (None, ""):
        images = expand_image_reference(source_ref)
        dataset_info = {"source": json_safe(source_ref), "split": None, "root": None, "names_count": 0}
        names: dict[int, str] = {}
    else:
        images, dataset_info, names = collect_dataset_images(data_ref, split)
    images = dedupe_images(images)
    limit_raw = evaluate_params.get("limit", evaluate_params.get("max_images", 5))
    limit = int(limit_raw) if limit_raw not in (None, "") else 5
    limit_value = None if limit <= 0 else limit
    seed_raw = evaluate_params.get("seed")
    seed = int(seed_raw) if seed_raw not in (None, "") else None
    sample = select_image_sample(
        images,
        limit=limit_value,
        offset=int(evaluate_params.get("offset", 0)),
        stride=max(1, int(evaluate_params.get("stride", 1))),
        shuffle=parse_bool(evaluate_params.get("shuffle"), False),
        seed=seed,
    )
    dataset_info["images_total"] = len(images)
    dataset_info["sample_count"] = len(sample)
    dataset_info["sample_limit"] = limit_value
    dataset_info["sample_offset"] = int(evaluate_params.get("offset", 0))
    dataset_info["sample_stride"] = max(1, int(evaluate_params.get("stride", 1)))
    if not sample:
        raise ValueError("No local images were found for multimodal evaluation.")
    return sample, dataset_info, names


def label_path_for_image(image_path: Path) -> Path:
    parts = image_path.resolve().parts
    if "images" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_ground_truth_summary(image_path: Path, names: dict[int, str], max_objects: int = 30) -> dict[str, Any]:
    label_path = label_path_for_image(image_path)
    summary: dict[str, Any] = {"path": str(label_path), "exists": label_path.exists(), "objects": 0, "labels": [], "label_counts": {}}
    if not label_path.exists():
        return summary
    labels: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xywhn = [round(float(value), 6) for value in parts[1:5]]
        except Exception:
            continue
        label = names.get(class_id, str(class_id))
        counts[label] = counts.get(label, 0) + 1
        labels.append({"class_id": class_id, "label": label, "xywhn": xywhn})
    summary["objects"] = len(labels)
    summary["labels"] = labels[:max_objects]
    summary["label_counts"] = counts
    if len(labels) > max_objects:
        summary["truncated"] = len(labels) - max_objects
    return summary


def merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        try:
            target[str(key)] = target.get(str(key), 0) + int(value)
        except Exception:
            continue


def detection_label_counts(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in detections:
        for detection in item.get("detections", []) or []:
            label = detection.get("label")
            if label is None:
                continue
            counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def preferred_verdict(item: dict[str, Any]) -> dict[str, Any]:
    multimodal = item.get("multimodal", {}) or {}
    llm = multimodal.get("llm_refine", {}) or {}
    if isinstance(llm, dict) and isinstance(llm.get("verdict"), dict):
        return llm["verdict"]
    vlm = multimodal.get("vlm", {}) or {}
    return vlm.get("verdict", {}) if isinstance(vlm, dict) and isinstance(vlm.get("verdict"), dict) else {}


def verdict_field_count(verdict: dict[str, Any], field: str) -> int:
    cross_check = verdict.get("yolo_cross_check", {}) if isinstance(verdict, dict) else {}
    if not isinstance(cross_check, dict):
        return 0
    value = cross_check.get(field)
    if isinstance(value, list):
        return len(value)
    return 1 if value else 0


def aggregate_multimodal_evaluation(items: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    gt_counts: dict[str, int] = {}
    detection_counts: dict[str, int] = {}
    flag_counts = {"confirmed": 0, "false_positives": 0, "possible_misses": 0, "duplicate_or_fragmented": 0}
    parsed = 0
    total_boxes = 0
    total_gt_objects = 0
    gt_available = 0
    for item in items:
        status = str(item.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        detector = item.get("detector", {}) or {}
        total_boxes += int(detector.get("boxes", 0) or 0)
        merge_counts(detection_counts, detector.get("label_counts", {}) or {})
        ground_truth = item.get("ground_truth", {}) or {}
        if ground_truth.get("exists"):
            gt_available += 1
            total_gt_objects += int(ground_truth.get("objects", 0) or 0)
            merge_counts(gt_counts, ground_truth.get("label_counts", {}) or {})
        multimodal = item.get("multimodal", {}) or {}
        vlm = multimodal.get("vlm", {}) or {}
        llm = multimodal.get("llm_refine", {}) or {}
        if vlm.get("verdict_parse_status") == "parsed" or llm.get("verdict_parse_status") == "parsed":
            parsed += 1
        verdict = preferred_verdict(item)
        for field in flag_counts:
            flag_counts[field] += verdict_field_count(verdict, field)
    total = len(items)
    return {
        "images_processed": total,
        "status_counts": status_counts,
        "verdicts_parsed": parsed,
        "verdict_parse_rate": round(parsed / total, 4) if total else 0.0,
        "detections_total": total_boxes,
        "ground_truth_total": total_gt_objects,
        "ground_truth_images": gt_available,
        "avg_detected_boxes": round(total_boxes / total, 4) if total else 0.0,
        "avg_ground_truth_objects": round(total_gt_objects / gt_available, 4) if gt_available else None,
        "detection_label_counts": detection_counts,
        "ground_truth_label_counts": gt_counts,
        "cross_check_flag_counts": flag_counts,
    }


def overall_multimodal_evaluation_status(aggregate: dict[str, Any]) -> str:
    counts = aggregate.get("status_counts", {}) or {}
    total = int(aggregate.get("images_processed", 0) or 0)
    if total == 0:
        return "failed"
    if counts.get("ok") == total:
        return "ok"
    if counts.get("blocked") == total:
        return "blocked"
    if counts.get("failed") == total:
        return "failed"
    return "partial"


def build_multimodal_evaluation_prompt(
    base_prompt: str,
    image_path: Path,
    index: int,
    total: int,
    ground_truth: dict[str, Any],
    *,
    include_ground_truth: bool,
) -> str:
    prompt = (
        f"{base_prompt}\n\n"
        f"Evaluation image {index + 1}/{total}: {image_path.name}. "
        "Focus on detector agreement, obvious false positives, likely missed objects, duplicates, and uncertainty."
    )
    if include_ground_truth:
        prompt += "\n\nGround-truth labels for post-hoc comparison:\n" + json.dumps(ground_truth, ensure_ascii=False, indent=2)
    return prompt


def multimodal_prompt_from_request(request: dict[str, Any], params: dict[str, Any]) -> str:
    return (
        request.get("inputs", {}).get("prompt")
        or params.get("prompt")
        or params.get("question")
        or "Explain the scene, verify the YOLO detections, and identify important missed or uncertain visual evidence."
    )


def run_multimodal_infer(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request["params"])
    yolo_params, multimodal_params = split_yolo_and_multimodal_params(params)
    source = request["inputs"].get("source") or yolo_params.pop("source", None) or multimodal_params.get("source")
    if source is None:
        raise ValueError("`inputs.source` is required for yolo.multimodal.infer.")
    prompt = multimodal_prompt_from_request(request, multimodal_params)
    provider_cfg = openai_config(multimodal_params)
    if provider_cfg["provider"] != "openai":
        raise ValueError(f"Unsupported multimodal provider: {provider_cfg['provider']}")

    max_items = int(yolo_params.pop("max_items", multimodal_params.get("max_reasoning_items", 3)))
    max_boxes = int(multimodal_params.get("max_reasoning_boxes", 20))
    thinking_with_image = parse_bool(multimodal_params.get("thinking_with_image"), True)
    structured_output = parse_bool(multimodal_params.get("structured_output"), True)
    max_output_tokens = int(multimodal_params.get("max_output_tokens", 1000 if structured_output else 800))
    device_selection = resolve_device_selection(request, yolo_params)
    yolo_params, chosen_device, auto_completed = apply_runtime_defaults(request, yolo_params, purpose="predict")
    effective_request = deepcopy(request)
    effective_request["params"] = yolo_params
    effective_request["inputs"]["source"] = source

    method = str(multimodal_params.get("method") or ("thinking-with-image" if thinking_with_image else "detector-text-reflection"))
    if is_dry_run(request):
        return plan_response(
            effective_request,
            "multimodal inference dry run prepared",
            "orchestrator",
            "yolo.multimodal.infer",
            params={
                "stages": [
                    {"name": "yolo_predict", "executor": "python_api", "target": "YOLO(...).predict", "params": yolo_params},
                    {
                        "name": "vlm_reasoning",
                        "executor": "openai.compatible",
                        "provider": provider_cfg["provider"],
                        "model": provider_cfg["vlm_model"],
                        "api_mode": provider_cfg["api_mode"],
                        "image": "input_image" if thinking_with_image else "text_only",
                        "method": method,
                        "structured_output": structured_output,
                    },
                    {
                        "name": "llm_refine",
                        "executor": "openai.compatible",
                        "provider": provider_cfg["provider"],
                        "model": provider_cfg["llm_model"],
                        "api_mode": provider_cfg["api_mode"],
                        "enabled": bool(provider_cfg.get("llm_model") or multimodal_params.get("enable_llm_refine")),
                    },
                ],
                "prompt": prompt,
            },
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
                "multimodal": {
                    "provider": provider_cfg["provider"],
                    "vlm_model": provider_cfg["vlm_model"],
                    "llm_model": provider_cfg["llm_model"],
                    "api_mode": provider_cfg["api_mode"],
                    "api_key_env": provider_cfg["api_key_env"],
                    "api_key_present": provider_cfg["api_key_present"],
                    "method": method,
                    "thinking_with_image": thinking_with_image,
                    "structured_output": structured_output,
                },
            },
        )

    detections: list[dict[str, Any]]
    save_dir = None
    yolo_error = None
    if bool(multimodal_params.get("skip_yolo", False)):
        detections = json_safe(request["inputs"].get("detections") or multimodal_params.get("detections") or [])
    else:
        try:
            model = build_model(effective_request)
            results = model.predict(source=source, **yolo_params)
        except Exception as exc:
            if device_selection["source"] == "auto" and chosen_device not in (None, "cpu"):
                retry_params = replace_cli_device(yolo_params, "cpu")
                effective_request["params"] = retry_params
                try:
                    model = build_model(effective_request)
                    results = model.predict(source=source, **retry_params)
                    yolo_params = retry_params
                    chosen_device = "cpu"
                    auto_completed["device"] = "cpu"
                    auto_completed["device_source"] = "recovery"
                except Exception:
                    raise exc
            else:
                raise
        save_dir = getattr(model.predictor, "save_dir", None)
        detections = summarize_results_for_reasoning(results, max_items=max_items, max_boxes=max_boxes)

    image_ref = image_source_for_openai(source, detections)
    vlm_result: dict[str, Any]
    llm_result: dict[str, Any] | None = None
    image_meta: dict[str, Any] = {
        "requested": image_ref,
        "thinking_with_image": thinking_with_image,
        "attached": False,
    }
    if thinking_with_image and image_ref is None:
        vlm_result = {
            "status": "blocked",
            "provider": "openai",
            "summary": "No image source was available for multimodal reasoning.",
        }
    elif not provider_cfg["api_key_present"]:
        vlm_result = {
            "status": "blocked",
            "provider": "openai",
            "summary": "OPENAI_API_KEY is not set; multimodal reasoning was skipped.",
            "api_key_env": provider_cfg["api_key_env"],
        }
    else:
        try:
            encoded_image = None
            if thinking_with_image and image_ref is not None:
                encoded_image = encode_image_reference_for_openai(
                    image_ref,
                    max_bytes=int(multimodal_params.get("max_image_bytes", 20_000_000)),
                )
                image_meta = {k: v for k, v in encoded_image.items() if k != "image_url"}
                image_meta["thinking_with_image"] = True
                image_meta["attached"] = True
            user_text = build_thinking_with_image_prompt(
                prompt,
                detections,
                method=method,
                thinking_with_image=thinking_with_image,
                structured_output=structured_output,
            )
            vlm_result = call_openai_compatible(
                model=str(provider_cfg["vlm_model"]),
                user_text=user_text,
                developer_text=str(
                    multimodal_params.get("developer_prompt")
                    or multimodal_params.get("system_prompt")
                    or "You are a careful visual reasoning assistant for YOLO-Master. Use the image and detector evidence, but only return concise evidence and uncertainty, not hidden chain-of-thought."
                ),
                image_url=encoded_image["image_url"] if encoded_image else None,
                image_detail=str(multimodal_params.get("image_detail", "auto")),
                base_url=provider_cfg["base_url"],
                api_mode=str(provider_cfg["api_mode"]),
                max_output_tokens=max_output_tokens,
                temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
            )
            vlm_result = attach_multimodal_verdict(vlm_result)
        except Exception as exc:
            vlm_result = {
                "status": "failed",
                "provider": "openai",
                "summary": "Failed to prepare or call VLM reasoning.",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    llm_enabled = bool(provider_cfg.get("llm_model") or multimodal_params.get("enable_llm_refine"))
    if llm_enabled and vlm_result.get("status") == "ok":
        llm_model = str(provider_cfg.get("llm_model") or provider_cfg["vlm_model"])
        refine_prompt = (
            "Refine this YOLO + VLM inference into a compact final answer. Do not add unsupported visual claims. "
            "Return exactly one JSON object without Markdown fences. Use these keys: answer, visual_evidence, "
            "yolo_cross_check, uncertainty, recommended_next_actions. In yolo_cross_check, include arrays named "
            "confirmed, false_positives, possible_misses, duplicate_or_fragmented, and notes when applicable.\n\n"
            f"User task:\n{prompt}\n\n"
            f"YOLO detection summary:\n{json.dumps(json_safe(detections), ensure_ascii=False, indent=2)}\n\n"
            f"VLM answer:\n{vlm_result.get('text', '')}\n\n"
            f"Parsed VLM verdict:\n{json.dumps(json_safe(vlm_result.get('verdict', {})), ensure_ascii=False, indent=2)}"
        )
        llm_result = call_openai_compatible(
            model=llm_model,
            user_text=refine_prompt,
            developer_text="You are a concise verifier. Return answer, evidence, uncertainty, and next actions.",
            base_url=provider_cfg["base_url"],
            api_mode=str(provider_cfg["api_mode"]),
            max_output_tokens=max_output_tokens,
            temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
        )
        llm_result = attach_multimodal_verdict(llm_result)

    overall_status = multimodal_overall_status(vlm_result, llm_result)
    summary_map = {
        "ok": "multimodal inference finished",
        "blocked": "YOLO inference finished, but multimodal reasoning was blocked",
        "partial": "YOLO inference finished; multimodal reasoning was incomplete",
    }
    summary = summary_map[overall_status]
    environment = collect_environment_report(effective_request, selected_device=chosen_device)
    payload = response(
        request["skill"],
        overall_status,
        summary,
        job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "python_api+openai", "device": chosen_device},
        results=detections,
        environment=environment,
        auto_completed=auto_completed,
        multimodal={
            "method": method,
            "thinking_with_image": thinking_with_image,
            "structured_output": structured_output,
            "provider": {
                "name": provider_cfg["provider"],
                "base_url": provider_cfg["base_url"],
                "api_mode": provider_cfg["api_mode"],
                "api_key_env": provider_cfg["api_key_env"],
                "api_key_present": provider_cfg["api_key_present"],
            },
            "image": image_meta,
            "prompt": prompt,
            "vlm": vlm_result,
            "llm_refine": llm_result or {"status": "skipped"},
        },
        next_actions=["yolo.predict", "yolo.val"],
    )
    if yolo_error:
        payload["yolo_error"] = yolo_error
    if save_dir:
        payload["artifacts"] = [{"kind": "directory", "path": str(Path(save_dir).resolve())}]
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_multimodal_evaluate(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request["params"])
    yolo_params, multimodal_params, evaluate_params = split_yolo_multimodal_evaluate_params(params)
    prompt = multimodal_prompt_from_request(request, multimodal_params)
    provider_cfg = openai_config(multimodal_params)
    if provider_cfg["provider"] != "openai":
        raise ValueError(f"Unsupported multimodal provider: {provider_cfg['provider']}")

    thinking_with_image = parse_bool(multimodal_params.get("thinking_with_image"), True)
    structured_output = parse_bool(multimodal_params.get("structured_output"), True)
    max_output_tokens = int(multimodal_params.get("max_output_tokens", 1000 if structured_output else 800))
    include_ground_truth = parse_bool(
        evaluate_params.get("include_ground_truth_in_prompt", evaluate_params.get("include_ground_truth")),
        False,
    )
    run_yolo_val = parse_bool(evaluate_params.get("run_yolo_val"), False)
    continue_on_error = parse_bool(evaluate_params.get("continue_on_error"), True)
    report_name = str(evaluate_params.get("report_name") or "multimodal-evaluation.json")
    data_ref_for_baseline = request["inputs"].get("data") or evaluate_params.get("data") or yolo_params.get("data")
    device_selection = resolve_device_selection(request, yolo_params)
    yolo_params, chosen_device, auto_completed = apply_runtime_defaults(request, yolo_params, purpose="predict")
    if "verbose" not in yolo_params:
        yolo_params["verbose"] = False
        auto_completed["verbose"] = False

    effective_request = deepcopy(request)
    effective_request["params"] = yolo_params
    if request["inputs"].get("data") is not None:
        effective_request["inputs"]["data"] = request["inputs"]["data"]
    if request["inputs"].get("source") is not None:
        effective_request["inputs"]["source"] = request["inputs"]["source"]

    images, dataset_info, names = collect_multimodal_evaluation_images(request, evaluate_params, yolo_params, multimodal_params)
    effective_request["inputs"]["source"] = str(images[0].parent)
    environment = collect_environment_report(effective_request, selected_device=chosen_device)

    if is_dry_run(request):
        return plan_response(
            effective_request,
            "multimodal evaluation dry run prepared",
            "orchestrator",
            "yolo.multimodal.evaluate",
            params={
                "stages": [
                    {"name": "collect_images", "executor": "dataset_or_source_resolver"},
                    {"name": "yolo_predict_batch", "executor": "python_api", "target": "YOLO(...).predict"},
                    {"name": "vlm_reasoning_batch", "executor": "openai.compatible"},
                    {"name": "aggregate_report", "executor": "python"},
                ],
                "sample_count": len(images),
                "sample_images": [str(path) for path in images[:10]],
                "split": dataset_info.get("split"),
                "run_yolo_val": run_yolo_val,
                "include_ground_truth_in_prompt": include_ground_truth,
                "prompt": prompt,
            },
            extra={
                "environment": environment,
                "auto_completed": auto_completed,
                "multimodal": {
                    "provider": provider_cfg["provider"],
                    "vlm_model": provider_cfg["vlm_model"],
                    "llm_model": provider_cfg["llm_model"],
                    "api_mode": provider_cfg["api_mode"],
                    "api_key_env": provider_cfg["api_key_env"],
                    "api_key_present": provider_cfg["api_key_present"],
                    "method": str(multimodal_params.get("method") or ("thinking-with-image" if thinking_with_image else "detector-text-reflection")),
                    "thinking_with_image": thinking_with_image,
                    "structured_output": structured_output,
                    "dataset": dataset_info,
                },
            },
        )

    model = build_model(effective_request)
    selected_device = chosen_device
    items: list[dict[str, Any]] = []
    save_dir = None
    method = str(multimodal_params.get("method") or ("thinking-with-image" if thinking_with_image else "detector-text-reflection"))
    llm_enabled = bool(provider_cfg.get("llm_model") or multimodal_params.get("enable_llm_refine"))

    for index, image_path in enumerate(images):
        image_item: dict[str, Any] = {
            "path": str(image_path),
            "index": index,
            "ground_truth": read_ground_truth_summary(image_path, names),
        }
        image_prompt = build_multimodal_evaluation_prompt(
            prompt,
            image_path,
            index,
            len(images),
            image_item["ground_truth"],
            include_ground_truth=include_ground_truth,
        )
        yolo_prediction = None
        try:
            yolo_prediction = model.predict(source=str(image_path), **yolo_params)
        except Exception as exc:
            if device_selection["source"] == "auto" and selected_device not in (None, "cpu") and request.get("runtime", {}).get("allow_device_fallback", True):
                retry_params = replace_cli_device(yolo_params, "cpu")
                effective_request["params"] = retry_params
                try:
                    model = build_model(effective_request)
                    yolo_prediction = model.predict(source=str(image_path), **retry_params)
                    yolo_params = retry_params
                    selected_device = "cpu"
                    auto_completed["device"] = "cpu"
                    auto_completed["device_source"] = "recovery"
                except Exception as retry_exc:
                    if not continue_on_error:
                        raise retry_exc
                    image_item["status"] = "failed"
                    image_item["error"] = {"type": type(retry_exc).__name__, "message": str(retry_exc)}
                    items.append(image_item)
                    continue
            elif continue_on_error:
                image_item["status"] = "failed"
                image_item["error"] = {"type": type(exc).__name__, "message": str(exc)}
                items.append(image_item)
                continue
            else:
                raise

        detections = summarize_results_for_reasoning(yolo_prediction, max_items=1, max_boxes=int(multimodal_params.get("max_reasoning_boxes", 20)))
        detection_summary = detections[0] if detections else {"path": str(image_path), "speed": {}, "detections": []}
        image_ref = image_source_for_openai(str(image_path), detections)
        image_meta: dict[str, Any] = {
            "requested": image_ref,
            "thinking_with_image": thinking_with_image,
            "attached": False,
        }
        vlm_result: dict[str, Any]
        llm_result: dict[str, Any] | None = None
        if thinking_with_image and image_ref is None:
            vlm_result = {
                "status": "blocked",
                "provider": "openai",
                "summary": "No image source was available for multimodal reasoning.",
            }
        elif not provider_cfg["api_key_present"]:
            vlm_result = {
                "status": "blocked",
                "provider": "openai",
                "summary": "OPENAI_API_KEY is not set; multimodal reasoning was skipped.",
                "api_key_env": provider_cfg["api_key_env"],
            }
        else:
            try:
                encoded_image = None
                if thinking_with_image and image_ref is not None:
                    encoded_image = encode_image_reference_for_openai(
                        image_ref,
                        max_bytes=int(multimodal_params.get("max_image_bytes", 20_000_000)),
                    )
                    image_meta = {k: v for k, v in encoded_image.items() if k != "image_url"}
                    image_meta["thinking_with_image"] = True
                    image_meta["attached"] = True
                user_text = build_thinking_with_image_prompt(
                    image_prompt,
                    detections,
                    method=method,
                    thinking_with_image=thinking_with_image,
                    structured_output=structured_output,
                )
                vlm_result = call_openai_compatible(
                    model=str(provider_cfg["vlm_model"]),
                    user_text=user_text,
                    developer_text=str(
                        multimodal_params.get("developer_prompt")
                        or multimodal_params.get("system_prompt")
                        or "You are a careful visual reasoning assistant for YOLO-Master. Use the image and detector evidence, but only return concise evidence and uncertainty, not hidden chain-of-thought."
                    ),
                    image_url=encoded_image["image_url"] if encoded_image else None,
                    image_detail=str(multimodal_params.get("image_detail", "auto")),
                    base_url=provider_cfg["base_url"],
                    api_mode=str(provider_cfg["api_mode"]),
                    max_output_tokens=max_output_tokens,
                    temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
                )
                vlm_result = attach_multimodal_verdict(vlm_result)
            except Exception as exc:
                vlm_result = {
                    "status": "failed",
                    "provider": "openai",
                    "summary": "Failed to prepare or call VLM reasoning.",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }

        if llm_enabled and vlm_result.get("status") == "ok":
            llm_model = str(provider_cfg.get("llm_model") or provider_cfg["vlm_model"])
            refine_prompt = (
                "Refine this YOLO + VLM evaluation into a compact final answer. Do not add unsupported visual claims. "
                "Return exactly one JSON object without Markdown fences. Use these keys: answer, visual_evidence, "
                "yolo_cross_check, uncertainty, recommended_next_actions. In yolo_cross_check, include arrays named "
                "confirmed, false_positives, possible_misses, duplicate_or_fragmented, and notes when applicable.\n\n"
                f"User task:\n{image_prompt}\n\n"
                f"YOLO detection summary:\n{json.dumps(json_safe(detections), ensure_ascii=False, indent=2)}\n\n"
                f"VLM answer:\n{vlm_result.get('text', '')}\n\n"
                f"Parsed VLM verdict:\n{json.dumps(json_safe(vlm_result.get('verdict', {})), ensure_ascii=False, indent=2)}"
            )
            llm_result = call_openai_compatible(
                model=llm_model,
                user_text=refine_prompt,
                developer_text="You are a concise verifier. Return answer, evidence, uncertainty, and next actions.",
                base_url=provider_cfg["base_url"],
                api_mode=str(provider_cfg["api_mode"]),
                max_output_tokens=max_output_tokens,
                temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
            )
            llm_result = attach_multimodal_verdict(llm_result)

        status = multimodal_overall_status(vlm_result, llm_result)
        image_item.update(
            {
                "status": status,
                "detector": {
                    "boxes": len(detection_summary.get("detections", []) or []),
                    "summary": detection_summary,
                    "label_counts": detection_label_counts(detections),
                },
                "prompt": image_prompt,
                "multimodal": {
                    "method": method,
                    "thinking_with_image": thinking_with_image,
                    "structured_output": structured_output,
                    "provider": {
                        "name": provider_cfg["provider"],
                        "base_url": provider_cfg["base_url"],
                        "api_mode": provider_cfg["api_mode"],
                        "api_key_env": provider_cfg["api_key_env"],
                        "api_key_present": provider_cfg["api_key_present"],
                    },
                    "image": image_meta,
                    "prompt": image_prompt,
                    "vlm": vlm_result,
                    "llm_refine": llm_result or {"status": "skipped"},
                },
            }
        )
        if status in {"blocked", "partial", "failed"}:
            image_item["notes"] = ["See multimodal cross-check and detection summary for details."]
        items.append(image_item)
        save_dir = getattr(getattr(model, "predictor", None), "save_dir", save_dir)

    aggregate = aggregate_multimodal_evaluation(items)
    overall_status = overall_multimodal_evaluation_status(aggregate)
    if run_yolo_val and data_ref_for_baseline not in (None, ""):
        baseline_request = normalize_request(
            {
                "skill": "yolo.val",
                "runtime": request.get("runtime", {}),
                "inputs": {"model": request["inputs"]["model"], "data": data_ref_for_baseline},
                "params": {k: v for k, v in yolo_params.items() if k not in {"source"}},
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": f"{request.get('request_id', default_request_id('yolo.multimodal.evaluate'))}-baseline",
            }
        )
        baseline = run_val(baseline_request)
    else:
        baseline = {"status": "skipped"}

    final_selection_source = "recovery" if selected_device != chosen_device and selected_device == "cpu" else device_selection["source"]
    environment = collect_environment_report(
        effective_request,
        selected_device=selected_device,
        requested_device=chosen_device,
        selection_source=final_selection_source,
    )

    report_dir = ensure_manifest_dir(request)
    report_path = report_dir / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "skill": request.get("skill"),
        "request_id": request.get("request_id"),
        "dataset": dataset_info,
        "aggregate": aggregate,
        "baseline": baseline,
        "items": items,
        "environment": environment,
        "auto_completed": auto_completed,
        "multimodal": {
            "method": method,
            "thinking_with_image": thinking_with_image,
            "structured_output": structured_output,
            "provider": {
                "name": provider_cfg["provider"],
                "base_url": provider_cfg["base_url"],
                "api_mode": provider_cfg["api_mode"],
                "api_key_env": provider_cfg["api_key_env"],
                "api_key_present": provider_cfg["api_key_present"],
            },
            "dataset": dataset_info,
            "prompt": prompt,
        },
    }
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = [{"kind": "json", "path": str(report_path.resolve())}]
    if save_dir:
        artifacts.append({"kind": "directory", "path": str(Path(save_dir).resolve())})

    payload = response(
        request["skill"],
        overall_status,
        f"multimodal evaluation finished on {len(items)} images",
        job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "python_api+openai", "device": selected_device},
        results=items,
        evaluation=aggregate,
        environment=environment,
        auto_completed=auto_completed,
        artifacts=artifacts,
        multimodal=report["multimodal"],
        baseline=baseline,
        next_actions=["yolo.val", "yolo.multimodal.infer", "yolo.predict"],
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_export(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request["params"])
    device_selection = resolve_device_selection(request, params)
    chosen_device = device_selection["device"]
    environment = collect_environment_report(
        request,
        selected_device=chosen_device,
        requested_device=chosen_device,
        selection_source=device_selection["source"],
    )
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
            return cli_plan(
                request,
                cli_args_from_values("export", values),
                extra={"environment": environment, "auto_completed": {}},
            )
        return plan_response(
            request,
            "export dry run prepared",
            "python_api",
            "YOLO(...).export",
            params=request["params"],
            extra={"environment": environment, "auto_completed": {}},
        )

    if prefer_cli(request):
        values = build_cli_key_values(request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
        cli_result = run_cli(cli_args_from_values("export", values))
        failed = ensure_cli_success(request, cli_result, "export failed")
        if failed:
            return failed
        save_dir = cli_save_dir(request, values)
        artifacts = []
        if save_dir and save_dir.exists():
            for candidate in sorted(save_dir.rglob("*")):
                if candidate.is_file() and candidate.suffix not in {".csv", ".yaml", ".json", ".txt", ".jpg", ".png"}:
                    artifacts.append({"kind": "exported_model", "path": str(candidate.resolve())})
        return response(
            request["skill"],
            "ok",
            "export finished",
            artifacts=artifacts,
            environment=collect_environment_report(
                request,
                selected_device=chosen_device,
                requested_device=chosen_device,
                selection_source=device_selection["source"],
                cli_info=cli_result["install"],
            ),
            auto_completed={},
            job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "cli"},
            logs=cli_logs(cli_result),
        )

    model = build_model(request)
    exported = model.export(**request["params"])
    payload = response(
        request["skill"],
        "ok",
        "export finished",
        artifacts=[{"kind": "exported_model", "path": str(Path(exported).resolve())}],
        environment=environment,
        auto_completed={},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_benchmark(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request["params"])
    data = request["inputs"].get("data") or params.pop("data", None)
    device_selection = resolve_device_selection(request, params)
    params, chosen_device, auto_completed = apply_runtime_defaults(request, params, purpose="benchmark")
    effective_request = deepcopy(request)
    effective_request["params"] = params
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
            return cli_plan(
                request,
                cli_args_from_values("benchmark", values),
                extra={
                    "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                    "auto_completed": auto_completed,
                },
            )
        return plan_response(
            request,
            "benchmark dry run prepared",
            "python_api",
            "YOLO(...).benchmark",
            params={"data": data, **params},
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
            },
        )

    if prefer_cli(request):
        values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
        cli_execution = run_cli_with_recovery(
            request,
            "benchmark",
            values,
            failure_summary="benchmark failed",
            selected_device=chosen_device,
            selection_source=device_selection["source"],
        )
        failed = cli_execution["failed"]
        if failed:
            return failed
        cli_result = cli_execution["cli_result"]
        values = cli_execution["values"]
        final_device = cli_execution["device"]
        recovery = cli_execution["recovery"]
        save_dir = cli_save_dir(request, values)
        return response(
            request["skill"],
            "ok",
            "benchmark finished after automatic cpu fallback" if recovery and recovery.get("recovered") else "benchmark finished",
            data={"benchmark": {}},
            environment=collect_environment_report(
                effective_request,
                selected_device=final_device,
                requested_device=chosen_device,
                selection_source="recovery" if recovery and recovery.get("recovered") else device_selection["source"],
                cli_info=cli_result["install"],
            ),
            auto_completed=auto_completed,
            job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "cli", "device": final_device},
            logs=cli_logs(cli_result),
            attempts=cli_execution["attempts"] if recovery else [],
            recovery=recovery or {},
        )

    model = build_model(request)
    benchmark_result = model.benchmark(data=data, **params)
    payload = response(
        request["skill"],
        "ok",
        "benchmark finished",
        data={"benchmark": json_safe(benchmark_result)},
        environment=collect_environment_report(effective_request, selected_device=chosen_device),
        auto_completed=auto_completed,
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_tune(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request["params"])
    use_ray = bool(params.pop("use_ray", False))
    iterations = int(params.pop("iterations", 10))
    if request["inputs"].get("data") and "data" not in params:
        params["data"] = request["inputs"]["data"]
    if is_dry_run(request):
        return plan_response(
            request,
            "tune dry run prepared",
            "python_api",
            "YOLO(...).tune",
            params={"use_ray": use_ray, "iterations": iterations, **params},
        )

    model = build_model(request)
    tuned = model.tune(use_ray=use_ray, iterations=iterations, **params)
    payload = response(request["skill"], "ok", "tuning finished", data={"tune": json_safe(tuned)})
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_lora_adapters(request: dict[str, Any]) -> dict[str, Any]:
    action = request.get("action") or request["params"].get("action")
    if not action:
        raise ValueError("`action` is required for yolo.lora.adapters.")
    params = dict(request["params"])
    path = request["inputs"].get("path") or params.get("path")
    if is_dry_run(request):
        return plan_response(
            request,
            "LoRA adapter dry run prepared",
            "python_api",
            f"YOLO(...).lora::{action}",
            params={"path": path, **params},
        )

    model = build_model(request)
    if action == "save":
        if not path:
            raise ValueError("`inputs.path` is required for adapter save.")
        ok = model.save_lora_only(path)
        payload = response(
            request["skill"],
            "ok" if ok else "failed",
            "adapter save finished" if ok else "adapter save skipped",
            artifacts=[{"kind": "adapter", "path": str(Path(path).resolve())}],
        )
    elif action == "load":
        if not path:
            raise ValueError("`inputs.path` is required for adapter load.")
        ok = model.load_lora(path, merge=bool(params.get("merge", False)))
        payload = response(request["skill"], "ok" if ok else "failed", "adapter load finished")
    elif action == "merge":
        ok = model.merge_lora()
        payload = response(request["skill"], "ok" if ok else "failed", "adapter merge finished")
    else:
        raise ValueError(f"Unsupported adapter action: {action}")
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_moe_diagnose(request: dict[str, Any]) -> dict[str, Any]:
    inputs = request["inputs"]
    params = request["params"]
    model_path = inputs.get("model")
    dataset = inputs.get("data") or params.get("data", "coco8.yaml")
    batch_size = int(params.get("batch_size", 1))
    verbose = bool(params.get("verbose", False))
    output_dir = Path(params.get("output_dir") or ensure_manifest_dir(request) / "moe_diagnose")
    if is_dry_run(request):
        return plan_response(
            request,
            "MoE diagnose dry run prepared",
            "module",
            "diagnose_model",
            params={"model_path": model_path, "dataset": dataset, "batch_size": batch_size, "verbose": verbose, "output_dir": str(output_dir)},
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnose_model = get_moe_helpers()["diagnose_model"]
    with pushd(output_dir):
        _, stdout, stderr = capture_output(diagnose_model, model_path, dataset, batch_size, verbose)
    artifacts = []
    for name in ("expert_usage_heatmap.png", "expert_usage_bar.png"):
        file = output_dir / name
        if file.exists():
            artifacts.append({"kind": "image", "path": str(file.resolve())})
    payload = response(
        request["skill"],
        "ok",
        "moe diagnosis finished",
        artifacts=artifacts,
        logs={"stdout": stdout, "stderr": stderr},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_moe_prune(request: dict[str, Any]) -> dict[str, Any]:
    inputs = request["inputs"]
    params = request["params"]
    model_path = inputs.get("model")
    dataset = inputs.get("data") or params.get("data", "coco8.yaml")
    output_path = params.get("output_path") or str(ensure_manifest_dir(request) / "pruned_model.pt")
    threshold = float(params.get("threshold", 0.15))
    if is_dry_run(request):
        return plan_response(
            request,
            "MoE prune dry run prepared",
            "module",
            "prune_moe_model",
            params={"model_path": model_path, "output_path": output_path, "threshold": threshold, "dataset": dataset},
        )

    prune_moe_model = get_moe_helpers()["prune_moe_model"]
    ok, stdout, stderr = capture_output(prune_moe_model, model_path, output_path, threshold, dataset)
    payload = response(
        request["skill"],
        "ok" if ok else "failed",
        "moe prune finished" if ok else "moe prune failed",
        artifacts=[{"kind": "checkpoint", "path": str(Path(output_path).resolve())}] if Path(output_path).exists() else [],
        logs={"stdout": stdout, "stderr": stderr},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def format_solution_arg(key: str, value: Any) -> str:
    if isinstance(value, str):
        return f"{key}={value}"
    return f"{key}={repr(value)}"


def run_solutions(request: dict[str, Any]) -> dict[str, Any]:
    solution = request["inputs"].get("solution")
    if not solution:
        raise ValueError("`inputs.solution` is required for yolo.solutions.run.")
    if is_dry_run(request):
        if prefer_cli(request):
            args = ["solutions", solution]
            for key in ("model", "source"):
                if request["inputs"].get(key) is not None:
                    args.append(kv_arg(key, request["inputs"][key]))
            for key, value in request["params"].items():
                if key != "action":
                    args.append(kv_arg(key, value))
            return cli_plan(request, args)
        args = [solution]
        for key in ("model", "source"):
            if request["inputs"].get(key) is not None:
                args.append(format_solution_arg(key, request["inputs"][key]))
        for key, value in request["params"].items():
            if key != "action":
                args.append(format_solution_arg(key, value))
        return plan_response(request, "solutions dry run prepared", "module", "handle_yolo_solutions", params={"args": args})

    if prefer_cli(request):
        args = ["solutions", solution]
        for key in ("model", "source"):
            if request["inputs"].get(key) is not None:
                args.append(kv_arg(key, request["inputs"][key]))
        for key, value in request["params"].items():
            if key != "action":
                args.append(kv_arg(key, value))
        cli_result = run_cli(args)
        failed = ensure_cli_success(request, cli_result, "solutions run failed")
        if failed:
            return failed
        return response(
            request["skill"],
            "ok",
            "solution run finished",
            logs=cli_logs(cli_result),
            artifacts=[{"kind": "directory", "path": str((REPO_ROOT / "runs" / "solutions").resolve())}],
        )

    _, stdout, stderr = capture_output(get_cfg_helpers()["handle_yolo_solutions"], [solution, *[format_solution_arg(k, v) for k, v in request["inputs"].items() if k != "solution"], *[format_solution_arg(k, v) for k, v in request["params"].items() if k != "action"]])
    payload = response(
        request["skill"],
        "ok",
        "solution run finished",
        logs={"stdout": stdout, "stderr": stderr},
        artifacts=[{"kind": "directory", "path": str((REPO_ROOT / "runs" / "solutions").resolve())}],
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_ui_launch(request: dict[str, Any]) -> dict[str, Any]:
    mode = request["inputs"].get("mode") or request["params"].get("mode", "gradio")
    if is_dry_run(request):
        cmd_preview = [sys.executable, "app.py"] if mode == "gradio" else ["yolo", "solutions", "inference", f"model={request['inputs'].get('model', 'yolo11n.pt')}"]
        return plan_response(
            request,
            "UI launch dry run prepared",
            "cli" if mode == "streamlit" else "subprocess",
            mode,
            params={"cmd": cmd_preview},
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{mode}.stdout.log"
    stderr_path = LOG_DIR / f"{mode}.stderr.log"
    stdout_handle = open(stdout_path, "ab")
    stderr_handle = open(stderr_path, "ab")
    if mode == "gradio":
        cmd = [sys.executable, "app.py"]
        url = request["params"].get("url", "http://127.0.0.1:7860")
    elif mode == "streamlit":
        model = request["inputs"].get("model") or "yolo11n.pt"
        yolo_path, _ = ensure_yolo_cli()
        cmd = [yolo_path, "solutions", "inference", f"model={model}"]
        url = request["params"].get("url", "http://127.0.0.1:8501")
    else:
        raise ValueError(f"Unsupported ui launch mode: {mode}")
    process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=stdout_handle, stderr=stderr_handle, env=repo_cli_env())
    payload = response(
        request["skill"],
        "running",
        f"{mode} launcher started",
        job={"mode": "async", "pid": process.pid, "url": url},
        logs={"stdout_path": str(stdout_path.resolve()), "stderr_path": str(stderr_path.resolve())},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_pipeline(request: dict[str, Any]) -> dict[str, Any]:
    params = request["params"]
    common_inputs = deepcopy(request["inputs"])
    current_model = common_inputs.get("model")

    if is_dry_run(request):
        return plan_response(
            request,
            "pipeline dry run prepared",
            "orchestrator",
            "yolo.pipeline.experiment",
            params=params,
            next_actions=["yolo.train", "yolo.val", "yolo.export", "yolo.benchmark"],
        )

    stages = {}
    if "train" in params:
        train_request = normalize_request(
            {
                "skill": "yolo.train",
                "runtime": request.get("runtime", {}),
                "inputs": {**common_inputs, "model": current_model},
                "params": params["train"],
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": request.get("request_id"),
            }
        )
        stages["train"] = run_train_like(train_request, "yolo.train")
        current_model = best_checkpoint(stages["train"]) or current_model

    if "val" in params:
        val_request = normalize_request(
            {
                "skill": "yolo.val",
                "runtime": request.get("runtime", {}),
                "inputs": {**common_inputs, "model": current_model},
                "params": params["val"],
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": request.get("request_id"),
            }
        )
        stages["val"] = run_val(val_request)

    if "export" in params:
        export_request = normalize_request(
            {
                "skill": "yolo.export",
                "runtime": request.get("runtime", {}),
                "inputs": {**common_inputs, "model": current_model},
                "params": params["export"],
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": request.get("request_id"),
            }
        )
        stages["export"] = run_export(export_request)

    if "benchmark" in params:
        benchmark_request = normalize_request(
            {
                "skill": "yolo.benchmark",
                "runtime": request.get("runtime", {}),
                "inputs": {**common_inputs, "model": current_model},
                "params": params["benchmark"],
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": request.get("request_id"),
            }
        )
        stages["benchmark"] = run_benchmark(benchmark_request)

    payload = response(
        request["skill"],
        "ok",
        "pipeline finished",
        stages=stages,
        best_checkpoint=current_model,
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


HANDLERS = {
    "yolo.system": run_system,
    "yolo.model.inspect": run_model_inspect,
    "yolo.train": lambda request: run_train_like(request, "yolo.train"),
    "yolo.lora.train": lambda request: run_train_like(request, "yolo.lora.train"),
    "yolo.val": run_val,
    "yolo.predict": lambda request: run_predict_like(request, "predict"),
    "yolo.track": lambda request: run_predict_like(request, "track"),
    "yolo.multimodal.infer": run_multimodal_infer,
    "yolo.multimodal.evaluate": run_multimodal_evaluate,
    "yolo.export": run_export,
    "yolo.benchmark": run_benchmark,
    "yolo.tune": run_tune,
    "yolo.lora.adapters": run_lora_adapters,
    "yolo.moe.diagnose": run_moe_diagnose,
    "yolo.moe.prune": run_moe_prune,
    "yolo.solutions.run": run_solutions,
    "yolo.ui.launch": run_ui_launch,
    "yolo.pipeline.experiment": run_pipeline,
}


def load_request(args: argparse.Namespace) -> dict[str, Any]:
    if args.request:
        return json.loads(Path(args.request).read_text(encoding="utf-8"))
    if args.json:
        return json.loads(args.json)
    stdin = sys.stdin.read().strip()
    if stdin:
        return json.loads(stdin)
    raise ValueError("Provide --request, --json, or JSON on stdin.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured dispatcher for the YOLO-Master agent skill.")
    parser.add_argument("--request", help="Path to a JSON request file.")
    parser.add_argument("--json", help="Inline JSON request.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    request: dict[str, Any] | None = None
    try:
        request = normalize_request(load_request(args))
        skill = request.get("skill")
        if skill not in HANDLERS:
            raise ValueError(f"Unsupported skill: {skill}")
        payload = HANDLERS[skill](request)
        payload = finalize_payload(request, payload)
    except Exception as exc:
        payload = response(
            request.get("skill", "unknown") if request else "unknown",
            "failed",
            str(exc),
            error={"type": type(exc).__name__, "traceback": traceback.format_exc()},
        )
        if request:
            payload = finalize_payload(request, payload)

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("status") in {"ok", "running", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
