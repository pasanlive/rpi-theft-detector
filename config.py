"""Centralized configuration for the RPi5 Theft Detection Pipeline.

All hardware paths, model parameters, and pipeline constants are defined here
to ensure a single source of truth across all modules.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

# ─── Camera Configuration ───────────────────────────────────────────────────
# Source options: "picam" (Pi Camera Module 2 via libcamerasrc), "rtsp", or "v4l2"
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "picam").lower()

RTSP_URI = os.environ.get(
    "RTSP_URI",
    "rtsp://admin:ODPEDI@192.168.0.102:554/h264/ch1/sub/av_stream"
)

# ─── Hailo NPU Configuration ────────────────────────────────────────────────
HEF_MODEL_PATH = str(MODEL_DIR / "yolov8s_pose.hef")
HAILO_POST_SO = os.environ.get(
    "HAILO_POST_SO",
    "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolov8pose_post.so"
)

# ─── Pose Estimation Constants ───────────────────────────────────────────────
NUM_KEYPOINTS = 17          # COCO skeleton keypoints
KEYPOINT_DIM = 3            # (X, Y, Confidence) per keypoint
FEATURE_DIM = NUM_KEYPOINTS * KEYPOINT_DIM  # 51

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# ─── Temporal Sequence Configuration ─────────────────────────────────────────
SEQUENCE_LENGTH = 30        # Frames in the sliding window
CONFIDENCE_THRESHOLD = 0.45 # Below this → LOCF imputation triggers

# ─── BiLSTM Model Configuration ─────────────────────────────────────────────
LSTM_INPUT_DIM = FEATURE_DIM   # 51
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
NUM_CLASSES = 2
ACTION_LABELS = ["normal", "theft"]
MODEL_WEIGHTS_PATH = str(MODEL_DIR / "pose_lstm.pth")

# ─── Pipeline Timing ─────────────────────────────────────────────────────────
INFERENCE_INTERVAL_SEC = 0.1   # 10 Hz BiLSTM polling rate
DETECTION_CONFIDENCE_MIN = 0.35 # Minimum person detection confidence

# ─── GStreamer Pipeline Tuning ───────────────────────────────────────────────
ENABLE_VIDEO_FEED = os.environ.get("ENABLE_VIDEO_FEED", "false").lower() in ["true", "1", "yes"]
GST_QUEUE_MAX_BUFFERS = 3
GST_RTSP_LATENCY_MS = 100
FRAME_WIDTH = 640              # Sub-stream resolution
FRAME_HEIGHT = 480

# ─── Alert / Webhook Configuration ───────────────────────────────────────────
ALERT_LOG_FILE = str(PROJECT_ROOT / "alerts.log")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
ALERT_COOLDOWN_SEC = 30.0      # Minimum seconds between alerts
THEFT_CONFIDENCE_THRESHOLD = 0.7  # Min softmax probability to trigger alert
