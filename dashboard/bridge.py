"""Thread-safe shared state between the inference pipeline and the dashboard.

The pipeline's consumer thread pushes classification results and the
ThreadBridge provides live keypoint data. The dashboard server reads
snapshots of this state via SSE at ~2 Hz.

No modifications to any existing pipeline module are required — this
bridge wraps around the existing ThreadBridge and alert_handler callbacks.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from config import NUM_KEYPOINTS, KEYPOINT_DIM, FEATURE_DIM

logger = logging.getLogger(__name__)


class DashboardBridge:
    """Centralized state container read by the dashboard SSE endpoint.

    Parameters
    ----------
    thread_bridge : ThreadBridge, optional
        Reference to the ingestion ThreadBridge. Used to snapshot the
        latest keypoint vector for skeleton visualization.
    """

    def __init__(self, thread_bridge: Optional[Any] = None) -> None:
        self._lock = threading.Lock()
        self._thread_bridge = thread_bridge
        self._start_time = time.time()

        # Latest inference result
        self._inference: dict[str, Any] = {
            "label": "—",
            "confidence": 0.0,
            "timestamp": 0.0,
            "frame_count": 0,
        }

        # Inference rate tracking
        self._inference_times: deque[float] = deque(maxlen=30)

        # Alert history (newest first for the dashboard feed)
        self._alerts: deque[dict[str, Any]] = deque(maxlen=200)

        # Pipeline status
        self._pipeline_running = False

        logger.info("DashboardBridge initialized.")

    # ─── Producer Methods (called from pipeline threads) ───────────────────

    def push_inference(self, label: str, confidence: float) -> None:
        """Record a new classification result from the BiLSTM consumer.

        Called from the ConsumerThread — must be fast and thread-safe.
        """
        now = time.time()
        frame_count = 0
        if self._thread_bridge is not None:
            frame_count = self._thread_bridge.frame_count

        with self._lock:
            self._inference = {
                "label": label,
                "confidence": round(confidence, 4),
                "timestamp": now,
                "frame_count": frame_count,
            }
            self._inference_times.append(now)

    def push_alert(
        self, event_type: str, message: str, confidence: float = 0.0
    ) -> None:
        """Record an alert or notification event."""
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "message": message,
            "confidence": round(confidence, 4),
        }
        with self._lock:
            self._alerts.appendleft(entry)

    def set_pipeline_running(self, running: bool) -> None:
        """Update pipeline running status."""
        self._pipeline_running = running
        event = "Pipeline started" if running else "Pipeline stopped"
        self.push_alert("info", event)

    # ─── Consumer Methods (called from dashboard server) ───────────────────

    def get_state(self) -> dict[str, Any]:
        """Return a complete dashboard state snapshot.

        This is called by the SSE endpoint at ~2 Hz. Returns all data
        needed to render the dashboard in a single JSON payload.
        """
        # Snapshot keypoints from the live thread bridge (no lock needed —
        # ThreadBridge uses GIL-atomic deque operations)
        keypoints: Optional[list[float]] = None
        if self._thread_bridge is not None:
            try:
                latest = self._thread_bridge.get_latest_frame()
                if latest is not None:
                    keypoints = latest.tolist()
            except Exception:
                pass

        with self._lock:
            # Calculate inference FPS from recent timestamps
            fps = 0.0
            if len(self._inference_times) >= 2:
                span = self._inference_times[-1] - self._inference_times[0]
                if span > 0:
                    fps = (len(self._inference_times) - 1) / span

            return {
                "inference": {
                    **self._inference,
                    "fps": round(fps, 1),
                    "keypoints": keypoints,
                },
                "alerts": list(self._alerts)[:50],  # Last 50 for the feed
                "pipeline": {
                    "running": self._pipeline_running,
                    "uptime": round(time.time() - self._start_time, 1),
                },
            }
