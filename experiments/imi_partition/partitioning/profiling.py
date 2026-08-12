from __future__ import annotations

import json
import os
import resource
import threading
import time
from pathlib import Path
from typing import Dict

import torch


def _read_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage) * (1024 if os.name != "darwin" else 1)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ProcessProfiler:
    def __init__(self, *, enabled: bool, sample_interval_s: float = 0.1) -> None:
        self.enabled = bool(enabled)
        self.sample_interval_s = max(float(sample_interval_s), 0.01)
        self.start_time_s = 0.0
        self.peak_cpu_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.cuda_devices: list[int] = []

    def _sample_cpu_rss(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            self.peak_cpu_rss_bytes = max(self.peak_cpu_rss_bytes, _read_rss_bytes())

    def start(self) -> None:
        if not self.enabled:
            return
        self.start_time_s = time.perf_counter()
        self.peak_cpu_rss_bytes = _read_rss_bytes()
        if torch.cuda.is_available():
            self.cuda_devices = list(range(torch.cuda.device_count()))
            for device_idx in self.cuda_devices:
                try:
                    torch.cuda.reset_peak_memory_stats(device_idx)
                except RuntimeError:
                    continue
        self._thread = threading.Thread(target=self._sample_cpu_rss, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, object]:
        if not self.enabled:
            return {}
        self.peak_cpu_rss_bytes = max(self.peak_cpu_rss_bytes, _read_rss_bytes())
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval_s * 2.0)
        cuda_peak: Dict[str, Dict[str, int]] = {}
        cuda_devices: Dict[str, Dict[str, object]] = {}
        for device_idx in self.cuda_devices:
            try:
                properties = torch.cuda.get_device_properties(device_idx)
                cuda_peak[str(device_idx)] = {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_idx)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_idx)),
                }
                cuda_devices[str(device_idx)] = {
                    "name": str(properties.name),
                    "total_memory_bytes": int(properties.total_memory),
                }
            except RuntimeError:
                continue
        return {
            "elapsed_s": time.perf_counter() - self.start_time_s,
            "peak_cpu_rss_bytes": int(self.peak_cpu_rss_bytes),
            "cuda_peak_bytes": cuda_peak,
            "cuda_devices": cuda_devices,
        }
