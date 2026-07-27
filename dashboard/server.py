"""Lightweight Flask dashboard server with Server-Sent Events.

Serves the single-page monitoring dashboard and streams real-time
pipeline + system metrics via SSE at ~2 Hz.

Runs in a daemon thread alongside the GStreamer main loop — never
blocks the inference pipeline.

Usage (standalone dev mode)::

    python -m dashboard.server

Production (from main.py)::

    from dashboard.server import start_dashboard
    start_dashboard(dash_bridge, port=5000)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from flask import Flask, Response, send_from_directory

from dashboard.bridge import DashboardBridge
from dashboard.metrics import MetricsCollector

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _make_placeholder_jpeg() -> bytes:
    """Generate a clean dark placeholder JPEG image."""
    try:
        import cv2
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (33, 16, 11)  # Dark slate background
        cv2.putText(
            img, "HIGH-PERFORMANCE MODE ENABLED", (80, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (52, 211, 153), 2
        )
        cv2.putText(
            img, "Video feed disabled for max NPU inference FPS & zero latency", (50, 260),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1
        )
        cv2.putText(
            img, "Pass --enable-video to re-enable live MJPEG feed", (110, 300),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 189, 56), 1
        )
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()
    except Exception:
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
            b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
            b"\x1c $.' \x1c\x1c(7(-./3131#&2821.3111\xff\xc0\x00\x0b\x08\x00\x01"
            b"\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01"
            b"\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06"
            b"\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
        )


PLACEHOLDER_JPEG = _make_placeholder_jpeg()


def create_app(
    bridge: DashboardBridge,
    metrics: Optional[MetricsCollector] = None,
) -> Flask:
    """Factory function to create the Flask dashboard app.

    Parameters
    ----------
    bridge : DashboardBridge
        Shared state container connected to the inference pipeline.
    metrics : MetricsCollector, optional
        System metrics collector. Created internally if not provided.
    """
    if metrics is None:
        metrics = MetricsCollector()

    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.config["PROPAGATE_EXCEPTIONS"] = True

    # ─── Routes ────────────────────────────────────────────────────────────

    @app.route("/")
    def index() -> Response:
        """Serve the dashboard HTML."""
        return send_from_directory(str(STATIC_DIR), "index.html")

    @app.route("/api/state")
    def api_state() -> Response:
        """Single JSON snapshot of the full dashboard state (for polling)."""
        state = _build_state(bridge, metrics)
        return Response(
            json.dumps(state, default=str),
            mimetype="application/json",
        )

    @app.route("/api/stream")
    def api_stream() -> Response:
        """Server-Sent Events endpoint for real-time dashboard updates.

        Pushes a full state snapshot every ~500ms. The browser reconnects
        automatically if the connection drops (built-in EventSource behavior).
        """
        def generate():
            while True:
                state = _build_state(bridge, metrics)
                payload = json.dumps(state, default=str)
                yield f"data: {payload}\n\n"
                time.sleep(0.5)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/api/alerts/clear", methods=["POST"])
    def clear_alerts() -> Response:
        """Clear the alert history."""
        bridge._alerts.clear()
        return Response(
            json.dumps({"status": "ok"}),
            mimetype="application/json",
        )

    @app.route("/api/video_feed")
    def video_feed() -> Response:
        """Multipart MJPEG video stream endpoint for live video feed on dashboard."""
        def generate():
            while True:
                jpeg = bridge.get_latest_jpeg()
                if jpeg is None:
                    jpeg = PLACEHOLDER_JPEG

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
                time.sleep(0.05)  # ~20 FPS

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _build_state(
    bridge: DashboardBridge,
    metrics: MetricsCollector,
) -> dict[str, Any]:
    """Merge pipeline state with system metrics into a single payload."""
    state = bridge.get_state()
    state["system"] = metrics.sample()
    return state


def start_dashboard(
    bridge: DashboardBridge,
    host: str = "0.0.0.0",
    port: int = 5000,
) -> threading.Thread:
    """Start the dashboard server in a background daemon thread.

    Parameters
    ----------
    bridge : DashboardBridge
        Shared state from the inference pipeline.
    host : str
        Bind address. ``0.0.0.0`` for network access.
    port : int
        HTTP port.

    Returns
    -------
    threading.Thread
        The daemon thread running the Flask server.
    """
    app = create_app(bridge)

    def _run() -> None:
        # Suppress Flask/Werkzeug request logs to avoid cluttering pipeline output
        werkzeug_log = logging.getLogger("werkzeug")
        werkzeug_log.setLevel(logging.WARNING)

        logger.info("Dashboard server starting on http://%s:%d", host, port)
        app.run(host=host, port=port, threaded=True, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True, name="DashboardServer")
    thread.start()
    return thread


# ─── Standalone dev mode ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bridge = DashboardBridge()
    bridge.set_pipeline_running(False)
    # Push some demo alerts
    bridge.push_alert("info", "Dashboard started in standalone mode")
    bridge.push_alert("warning", "No pipeline connected — showing demo data")
    app = create_app(bridge)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
