from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_workload_benchmark import __version__
from llm_workload_benchmark.config import BenchmarkConfig


def create_run(
    config: BenchmarkConfig,
    config_path: Path,
    *,
    project_root: Path | None = None,
    now: datetime | None = None,
    run_directory: Path | None = None,
) -> Path:
    """Create a run directory and write its reproducibility manifest."""
    root = (project_root or Path.cwd()).resolve()
    created_at = now or datetime.now(UTC)
    if run_directory is None:
        run_id = _new_run_id(created_at)
        output_root = _resolve_from_root(root, config.benchmark.output_root)
        resolved_run_directory = output_root / run_id
    else:
        resolved_run_directory = _resolve_from_root(root, run_directory)
        run_id = resolved_run_directory.name
    resolved_run_directory.mkdir(parents=True, exist_ok=False)

    resolved_config_path = config_path.resolve()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "project_version": __version__,
        "config_source": {
            "path": _display_path(resolved_config_path, root),
            "sha256": _sha256(resolved_config_path),
        },
        "config": config.model_dump(mode="json"),
        "environment": _environment_details(),
        "git": _git_details(root),
    }
    _write_json_atomically(resolved_run_directory / "manifest.json", manifest)
    return resolved_run_directory


def _new_run_id(created_at: datetime) -> str:
    timestamp = created_at.astimezone(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _resolve_from_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_details() -> dict[str, Any]:
    uname = platform.uname()
    memory_bytes = _physical_memory_bytes()
    return {
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "operating_system_version": platform.version(),
        "machine": platform.machine(),
        "machine_model": _sysctl_value("hw.model"),
        "processor": _sysctl_value("machdep.cpu.brand_string")
        or uname.processor
        or None,
        "physical_memory_bytes": memory_bytes,
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "llama_cpp_python_version": _package_version("llama-cpp-python"),
        "power_source": _power_source(),
        "graphics": _graphics_details(),
        "sensor_capabilities": {
            "process_cpu_utilization": "available",
            "process_memory": "available",
            "apple_gpu_utilization": (
                "available" if platform.system() == "Darwin" else "unsupported"
            ),
            "temperature_and_power": (
                "requires_root_powermetrics"
                if platform.system() == "Darwin"
                else "unsupported"
            ),
        },
    }


def _physical_memory_bytes() -> int | None:
    macos_memory = _sysctl_value("hw.memsize")
    if macos_memory is not None:
        try:
            return int(macos_memory)
        except ValueError:
            return None

    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _power_source() -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    return first_line.strip() or None


def _graphics_details() -> list[dict[str, Any]]:
    if platform.system() != "Darwin":
        return []
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    displays = payload.get("SPDisplaysDataType")
    if not isinstance(displays, list):
        return []
    return [
        {
            "name": display.get("sppci_model"),
            "vendor": display.get("spdisplays_vendor"),
            "cores": display.get("sppci_cores"),
            "metal_support": display.get("spdisplays_metal"),
        }
        for display in displays
        if isinstance(display, dict)
    ]


def _sysctl_value(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_details(root: Path) -> dict[str, Any]:
    commit = _run_git(root, "rev-parse", "HEAD")
    if commit is None:
        return {"available": False, "commit": None, "dirty": None}

    status = _run_git(root, "status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
    }


def _run_git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _write_json_atomically(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
