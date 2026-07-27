"""Alert dispatch for the RPi5 Theft Detection Pipeline.

Supports file logging and HTTP webhook push notifications.
Includes cooldown logic to prevent alert flooding.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import (
    ALERT_COOLDOWN_SEC,
    ALERT_LOG_FILE,
    THEFT_CONFIDENCE_THRESHOLD,
    WEBHOOK_URL,
)

logger = logging.getLogger(__name__)


class AlertHandler:
    """Dispatches theft alerts via file log and optional webhook.
    
    Implements cooldown to prevent rapid-fire alert flooding when the
    classifier oscillates near the decision boundary.
    """

    def __init__(
        self,
        log_file: str = ALERT_LOG_FILE,
        webhook_url: str = WEBHOOK_URL,
        cooldown_sec: float = ALERT_COOLDOWN_SEC,
        confidence_threshold: float = THEFT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._log_path = Path(log_file)
        self._webhook_url = webhook_url
        self._cooldown_sec = cooldown_sec
        self._confidence_threshold = confidence_threshold
        self._last_alert_time: float = 0.0
        
        # Ensure log directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("AlertHandler initialized — log=%s, webhook=%s",
                     self._log_path, "enabled" if webhook_url else "disabled")

    def on_classification(self, label: str, confidence: float) -> None:
        """Callback for ConsumerThread. Triggers alert if theft detected."""
        if label != "theft" or confidence < self._confidence_threshold:
            return
        
        now = time.monotonic()
        if (now - self._last_alert_time) < self._cooldown_sec:
            logger.debug("Alert suppressed — cooldown active (%.1fs remaining)",
                         self._cooldown_sec - (now - self._last_alert_time))
            return
        
        self._last_alert_time = now
        timestamp = datetime.now(timezone.utc).isoformat()
        
        self._log_to_file(timestamp, label, confidence)
        if self._webhook_url:
            self._send_webhook(timestamp, label, confidence)

    def _log_to_file(self, timestamp: str, label: str, confidence: float) -> None:
        """Append structured JSON alert to log file."""
        entry = {
            "timestamp": timestamp,
            "event": "theft_detected",
            "label": label,
            "confidence": round(confidence, 4),
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.warning("🚨 THEFT DETECTED — confidence=%.2f, logged to %s",
                           confidence, self._log_path)
        except OSError as exc:
            logger.error("Failed to write alert log: %s", exc)

    def _send_webhook(self, timestamp: str, label: str, confidence: float) -> None:
        """Send JSON POST to configured webhook URL."""
        payload = json.dumps({
            "timestamp": timestamp,
            "event": "theft_detected",
            "source": "rpi5-theft-detector",
            "label": label,
            "confidence": round(confidence, 4),
        }).encode("utf-8")
        
        req = urllib.request.Request(
            self._webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.info("Webhook sent — status=%d", resp.status)
        except (urllib.error.URLError, OSError) as exc:
            logger.error("Webhook failed: %s", exc)
