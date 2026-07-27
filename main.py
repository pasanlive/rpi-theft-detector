"""RPi5 Theft Detector — Main Entrypoint.

Orchestrates the full pipeline:
  1. Initialize ThreadBridge (lock-free deque)
  2. Initialize ActionClassifier (BiLSTM model)
  3. Initialize AlertHandler (file log + webhook)
  4. Initialize DashboardBridge + start dashboard server
  5. Start ConsumerThread (polls bridge → classifies → alerts + dashboard)
  6. Initialize IngestionEngine (GStreamer + Hailo)
  7. Enter GStreamer main loop (blocks on main thread)

Usage::

    python main.py
    python main.py --no-dashboard    # Skip dashboard server

    Dashboard available at http://<rpi-ip>:5000

Environment variables (optional overrides):
    RTSP_URI         — Camera stream URL
    HAILO_POST_SO    — Path to Hailo post-processing .so
    WEBHOOK_URL      — HTTP endpoint for theft alerts
    DASHBOARD_PORT   — Dashboard server port (default: 5000)
    GST_DEBUG        — GStreamer debug level (e.g., "2" for warnings)
"""

from __future__ import annotations

import logging
import os
import signal
import sys

from config import INFERENCE_INTERVAL_SEC, MODEL_WEIGHTS_PATH, CAMERA_SOURCE

# ─── Logging Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-22s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def main() -> int:
    """Initialize all pipeline stages and enter the main loop.

    Returns
    -------
    int
        Exit code: 0 on clean shutdown, 1 on error.
    """
    enable_dashboard = "--no-dashboard" not in sys.argv
    dashboard_port = int(os.environ.get("DASHBOARD_PORT", "5000"))

    # Determine camera source (CLI flag > ENV var > default)
    camera_source = CAMERA_SOURCE
    if "--picam" in sys.argv or "--camera" in sys.argv:
        camera_source = "picam"
    elif "--rtsp" in sys.argv:
        camera_source = "rtsp"
    elif "--v4l2" in sys.argv or "--usb" in sys.argv:
        camera_source = "v4l2"

    logger.info("═══════════════════════════════════════════════════════════")
    logger.info("  RPi5 Theft Detector — Zero-Copy Video Analytics Pipeline")
    logger.info("  Camera Source: %s", camera_source.upper())
    logger.info("═══════════════════════════════════════════════════════════")

    # ── Phase 2: Thread Bridge ─────────────────────────────────────────────
    from thread_manager import ThreadBridge, ConsumerThread

    bridge = ThreadBridge()
    logger.info("ThreadBridge initialized (maxlen=%d)", bridge._deque.maxlen)

    # ── Phase 3: Action Classifier ─────────────────────────────────────────
    from action_classifier import ActionClassifier

    classifier = ActionClassifier(model_path=MODEL_WEIGHTS_PATH)
    logger.info("ActionClassifier loaded.")

    # ── Alert Handler ──────────────────────────────────────────────────────
    from alert_handler import AlertHandler

    alert_handler = AlertHandler()

    # ── Dashboard Bridge & Server ──────────────────────────────────────────
    dash_bridge = None
    if enable_dashboard:
        from dashboard.bridge import DashboardBridge
        from dashboard.server import start_dashboard

        dash_bridge = DashboardBridge(thread_bridge=bridge)

        start_dashboard(
            bridge=dash_bridge,
            host="0.0.0.0",
            port=dashboard_port,
        )
        logger.info(
            "Dashboard server started on http://0.0.0.0:%d", dashboard_port,
        )

    # ── Composite Result Callback ──────────────────────────────────────────
    def _on_classification(label: str, confidence: float) -> None:
        alert_handler.on_classification(label, confidence)
        if dash_bridge is not None:
            dash_bridge.push_inference(label, confidence)
            if label == "theft" and confidence >= 0.7:
                dash_bridge.push_alert(
                    "theft",
                    f"THEFT DETECTED — confidence {confidence:.1%}",
                    confidence,
                )

    # ── Consumer Thread ────────────────────────────────────────────────────
    consumer = ConsumerThread(
        bridge=bridge,
        classifier_fn=classifier.predict,
        result_callback=_on_classification,
        interval=INFERENCE_INTERVAL_SEC,
    )
    consumer.start()
    logger.info(
        "ConsumerThread started (polling at %.0f Hz).",
        1.0 / INFERENCE_INTERVAL_SEC,
    )

    # ── Phase 1: GStreamer Ingestion ───────────────────────────────────────
    from ingestion_engine import IngestionEngine

    engine = IngestionEngine(
        bridge=bridge,
        camera_source=camera_source,
        dash_bridge=dash_bridge,
    )

    # ── Signal Handlers for Graceful Shutdown ──────────────────────────────
    def _shutdown_handler(signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down gracefully...", sig_name)
        consumer.stop()
        engine.stop()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Mark pipeline as running for the dashboard
    if dash_bridge is not None:
        dash_bridge.set_pipeline_running(True)

    # ── Build & Start Pipeline (blocks here) ───────────────────────────────
    try:
        engine.build_pipeline()
        logger.info("Starting GStreamer main loop (Ctrl+C to stop)...")
        engine.start()  # Blocks until EOS / error / signal
    except RuntimeError as exc:
        logger.error("Pipeline failed: %s", exc)
        consumer.stop()
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        consumer.stop()
        if dash_bridge is not None:
            dash_bridge.set_pipeline_running(False)
        logger.info("All threads stopped. Exiting.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
