"""System metrics collector for the RPi5 dashboard.

Reads CPU, memory, disk, and thermal data using psutil and
Linux sysfs thermal zones. Gracefully degrades on non-RPi systems
(returns placeholder values for development).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Optional psutil import ───────────────────────────────────────────────────

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not installed — system metrics will be unavailable.")
    PSUTIL_AVAILABLE = False

# ─── Thermal zone paths (Linux / Raspberry Pi) ────────────────────────────────

THERMAL_ZONES = [
    Path("/sys/class/thermal/thermal_zone0/temp"),  # CPU (RPi5)
    Path("/sys/class/thermal/thermal_zone1/temp"),  # GPU (RPi5, if available)
]


def _read_thermal_zone(path: Path) -> float | None:
    """Read temperature from a Linux thermal zone sysfs file.

    Returns temperature in °C, or None if unavailable.
    """
    try:
        raw = path.read_text().strip()
        return int(raw) / 1000.0  # millidegrees → degrees
    except (FileNotFoundError, ValueError, PermissionError):
        return None


class MetricsCollector:
    """Collects system resource metrics with history for sparkline charts.

    Maintains rolling 60-sample history for CPU and temperature,
    suitable for rendering ~60-second sparkline charts on the dashboard.
    """

    def __init__(self, history_len: int = 60) -> None:
        self._cpu_history: deque[float] = deque(maxlen=history_len)
        self._temp_history: deque[float] = deque(maxlen=history_len)
        self._last_sample_time: float = 0.0
        logger.info(
            "MetricsCollector initialized (psutil=%s, history=%d)",
            PSUTIL_AVAILABLE, history_len,
        )

    def sample(self) -> dict[str, Any]:
        """Collect a single metrics snapshot.

        Returns a dict with cpu, memory, disk, temperature, and history.
        Safe to call from any thread at any frequency.
        """
        now = time.time()
        result: dict[str, Any] = {
            "timestamp": now,
            "cpu": self._get_cpu(),
            "memory": self._get_memory(),
            "disk": self._get_disk(),
            "temperature": self._get_temperature(),
            "cpu_history": list(self._cpu_history),
            "temp_history": list(self._temp_history),
        }

        # Update history (throttle to at most 1 sample/sec)
        if now - self._last_sample_time >= 1.0:
            self._cpu_history.append(result["cpu"]["percent"])
            temp = result["temperature"]["cpu"]
            if temp is not None:
                self._temp_history.append(temp)
            self._last_sample_time = now

        return result

    def _get_cpu(self) -> dict[str, Any]:
        """CPU usage and core count."""
        if not PSUTIL_AVAILABLE:
            return {"percent": 0.0, "cores": 0, "freq_mhz": 0}

        return {
            "percent": psutil.cpu_percent(interval=None),
            "cores": psutil.cpu_count(logical=True) or 0,
            "freq_mhz": round(
                (psutil.cpu_freq().current if psutil.cpu_freq() else 0), 0
            ),
        }

    def _get_memory(self) -> dict[str, Any]:
        """RAM usage in bytes and percentage."""
        if not PSUTIL_AVAILABLE:
            return {"used": 0, "total": 0, "percent": 0.0}

        mem = psutil.virtual_memory()
        return {
            "used": mem.used,
            "total": mem.total,
            "percent": mem.percent,
        }

    def _get_disk(self) -> dict[str, Any]:
        """Root filesystem usage."""
        if not PSUTIL_AVAILABLE:
            return {"used": 0, "total": 0, "percent": 0.0}

        disk = psutil.disk_usage("/")
        return {
            "used": disk.used,
            "total": disk.total,
            "percent": disk.percent,
        }

    def _get_temperature(self) -> dict[str, float | None]:
        """CPU and GPU temperatures from thermal zones."""
        cpu_temp = _read_thermal_zone(THERMAL_ZONES[0])
        gpu_temp = _read_thermal_zone(THERMAL_ZONES[1])

        # Fallback: try psutil sensors
        if cpu_temp is None and PSUTIL_AVAILABLE:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # RPi uses 'cpu_thermal', x86 uses 'coretemp'
                    for key in ("cpu_thermal", "cpu-thermal", "coretemp"):
                        if key in temps and temps[key]:
                            cpu_temp = temps[key][0].current
                            break
            except (AttributeError, OSError):
                pass

        return {"cpu": cpu_temp, "gpu": gpu_temp}
