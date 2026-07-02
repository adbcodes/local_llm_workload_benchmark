from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class RuntimeTelemetry:
    """Sample comparable per-run process and hardware metrics."""

    output_path: Path
    interval_seconds: float = 1.0
    _samples: list[dict[str, Any]] = field(default_factory=list, init=False)
    _peak_rss_bytes: int | None = field(default=None, init=False)
    _active_item_peak_rss_bytes: int | None = field(default=None, init=False)
    _started_at: float = field(default=0.0, init=False)
    _cpu_started: float = field(default=0.0, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._cpu_started = time.process_time()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="runtime-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._sample()
        self._write_samples()
        elapsed = max(0.0, time.perf_counter() - self._started_at)
        cpu_seconds = max(0.0, time.process_time() - self._cpu_started)
        return self._summary(elapsed, cpu_seconds)

    def peak_rss_bytes(self) -> int | None:
        """Return the run-wide sampled peak RSS."""
        current = _current_process_rss_bytes()
        self._record_rss(current)
        return self._peak_rss_bytes

    def current_rss_bytes(self) -> int | None:
        """Return current RSS, without presenting a lifetime peak as current."""
        current = _current_process_rss_bytes()
        self._record_rss(current)
        return current

    def begin_item(self) -> int | None:
        """Start a new item-specific RSS window and return its initial RSS."""
        current = _current_process_rss_bytes()
        self._active_item_peak_rss_bytes = current
        self._record_rss(current)
        return current

    def end_item(self) -> int | None:
        """Close the active RSS window and return that item's sampled peak."""
        current = _current_process_rss_bytes()
        self._record_rss(current)
        peak = self._active_item_peak_rss_bytes
        self._active_item_peak_rss_bytes = None
        return peak

    def _record_rss(self, rss: int | None) -> None:
        if rss is None:
            return
        self._peak_rss_bytes = max(self._peak_rss_bytes or 0, rss)
        if self._active_item_peak_rss_bytes is not None:
            self._active_item_peak_rss_bytes = max(
                self._active_item_peak_rss_bytes, rss
            )

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        rss = _current_process_rss_bytes()
        self._record_rss(rss)
        gpu = _apple_gpu_metrics() if platform.system() == "Darwin" else {}
        power = _privileged_power_metrics() if _can_sample_powermetrics() else {}
        self._samples.append(
            {
                "elapsed_seconds": max(0.0, time.perf_counter() - self._started_at),
                "process_rss_bytes": rss,
                "process_cpu_seconds": max(0.0, time.process_time() - self._cpu_started),
                **gpu,
                **power,
            }
        )

    def _write_samples(self) -> None:
        with self.output_path.open("w", encoding="utf-8") as output:
            for sample in self._samples:
                output.write(json.dumps(sample, sort_keys=True) + "\n")

    def _summary(self, elapsed: float, cpu_seconds: float) -> dict[str, Any]:
        return {
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": len(self._samples),
            "elapsed_seconds": elapsed,
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_utilization_percent": (
                cpu_seconds / elapsed * 100 if elapsed > 0 else None
            ),
            "peak_sampled_process_rss_bytes": self._peak_rss_bytes,
            "mean_process_rss_bytes": _mean_field(self._samples, "process_rss_bytes"),
            "mean_system_gpu_utilization_percent": _mean_field(
                self._samples, "system_gpu_utilization_percent"
            ),
            "peak_system_gpu_utilization_percent": _max_field(
                self._samples, "system_gpu_utilization_percent"
            ),
            "mean_gpu_power_watts": _mean_field(self._samples, "gpu_power_watts"),
            "mean_cpu_power_watts": _mean_field(self._samples, "cpu_power_watts"),
            "mean_system_power_watts": _mean_field(
                self._samples, "system_power_watts"
            ),
            "mean_cpu_temperature_c": _mean_field(
                self._samples, "cpu_temperature_c"
            ),
            "sensor_status": {
                "process_cpu": "available",
                "process_memory": "available" if self._peak_rss_bytes else "unavailable",
                "apple_gpu": (
                    "available"
                    if any(
                        sample.get("system_gpu_utilization_percent") is not None
                        for sample in self._samples
                    )
                    else "unavailable"
                ),
                "temperature_and_power": (
                    "available"
                    if _can_sample_powermetrics()
                    else "requires_root_powermetrics"
                    if platform.system() == "Darwin" and shutil.which("powermetrics")
                    else "unsupported"
                ),
            },
        }


def _current_process_rss_bytes() -> int | None:
    if platform.system() == "Linux":
        try:
            pages = int(Path("/proc/self/statm").read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError, IndexError, OSError, ValueError):
            pass
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return int(result.stdout.strip()) * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak <= 0:
            return None
        return int(peak if platform.system() == "Darwin" else peak * 1024)


def _apple_gpu_metrics() -> dict[str, float | int | None]:
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return {
        "system_gpu_utilization_percent": _regex_number(
            result.stdout, r'"Device Utilization %"=(\d+(?:\.\d+)?)'
        ),
        "renderer_utilization_percent": _regex_number(
            result.stdout, r'"Renderer Utilization %"=(\d+(?:\.\d+)?)'
        ),
        "tiler_utilization_percent": _regex_number(
            result.stdout, r'"Tiler Utilization %"=(\d+(?:\.\d+)?)'
        ),
        "gpu_allocated_system_memory_bytes": _regex_integer(
            result.stdout, r'"Alloc system memory"=(\d+)'
        ),
    }


def _can_sample_powermetrics() -> bool:
    return (
        platform.system() == "Darwin"
        and shutil.which("powermetrics") is not None
        and hasattr(os, "geteuid")
        and os.geteuid() == 0
    )


def _privileged_power_metrics() -> dict[str, float | None]:
    try:
        result = subprocess.run(
            [
                "powermetrics",
                "-n",
                "1",
                "-i",
                "100",
                "--samplers",
                "cpu_power,gpu_power,thermal",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    text = result.stdout + result.stderr
    return {
        "cpu_power_watts": _milliwatts_to_watts(
            _regex_number(text, r"CPU Power:\s*([\d.]+)\s*mW")
        ),
        "gpu_power_watts": _milliwatts_to_watts(
            _regex_number(text, r"GPU Power:\s*([\d.]+)\s*mW")
        ),
        "system_power_watts": _milliwatts_to_watts(
            _regex_number(text, r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")
        ),
        "cpu_temperature_c": _regex_number(
            text, r"CPU die temperature:\s*([\d.]+)\s*C"
        ),
    }


def _regex_number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _regex_integer(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _milliwatts_to_watts(value: float | None) -> float | None:
    return value / 1000 if value is not None else None


def _mean_field(samples: list[dict[str, Any]], name: str) -> float | None:
    values = [sample[name] for sample in samples if isinstance(sample.get(name), int | float)]
    return mean(values) if values else None


def _max_field(samples: list[dict[str, Any]], name: str) -> float | None:
    values = [sample[name] for sample in samples if isinstance(sample.get(name), int | float)]
    return max(values) if values else None
