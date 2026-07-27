# RPi5 Theft Detector — Zero-Copy Video Analytics Pipeline

A highly concurrent, zero-copy video analytics system for **Raspberry Pi 5** with **Hailo-8 AI HAT+** (26 TOPS). Chains hardware-accelerated H.264 decode → NPU pose estimation → CPU BiLSTM action recognition to detect theft in real time.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RTSP Camera (H.264)                                                          │
│       │                                                                        │
│       ▼                                                                        │
│  ┌──────────┐     ┌───────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │ rtspsrc   │────▶│v4l2h264dec│────▶│  hailonet    │────▶│  hailofilter    │  │
│  │ (network) │     │ (HW dec)  │     │ (NPU infer)  │     │ (post-process)  │  │
│  └──────────┘     └───────────┘     └──────────────┘     └────────┬────────┘  │
│                                                                    │           │
│                              GStreamer Pipeline                     │           │
├────────────────────────────────────────────────────────────────────┼───────────┤
│                                                                    │           │
│                   ┌────────────────────┐                           │           │
│                   │    appsink         │◀──────────────────────────┘           │
│                   │  (metadata only)   │                                       │
│                   └────────┬───────────┘                                       │
│                            │ 51-dim keypoint vector                            │
│                            ▼                                                   │
│                   ┌────────────────────┐                                       │
│                   │   ThreadBridge     │  ◀── lock-free deque (maxlen=30)      │
│                   │ (Producer-Consumer)│                                       │
│                   └────────┬───────────┘                                       │
│                            │ [30, 51] sequence                                 │
│                            ▼                                                   │
│                   ┌────────────────────┐     ┌──────────────────┐             │
│                   │  LOCF Cleaning     │────▶│  BiLSTM (CPU)    │             │
│                   │  (data imputation) │     │  [1,30,51]→[1,2] │             │
│                   └────────────────────┘     └────────┬─────────┘             │
│                                                       │                        │
│                                              ┌────────▼─────────┐             │
│                                              │  AlertHandler    │             │
│                                              │  (log + webhook) │             │
│                                              └──────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

| Constraint | Solution |
|---|---|
| **No frame-to-userspace copy** | appsink reads only Hailo ROI metadata, never the image buffer |
| **Clock rate mismatch** (NPU 30fps vs CPU 10Hz) | Lock-free `collections.deque` bridge; `leaky=downstream` queues |
| **RTSP jitter** | `drop-on-latency=true`, `max-buffers=1`, `sync=false` |
| **Keypoint occlusion** | LOCF imputation for confidence < 0.45 |
| **Alert flooding** | 30-second cooldown between webhook/log dispatches |

## System Requirements

### Hardware
- Raspberry Pi 5 (ARM64, 4GB+ RAM)
- Hailo-8 AI HAT+ (26 TOPS, PCIe Gen 3.0)
- IP camera with RTSP H.264 output

### Software Prerequisites

#### System Packages (apt)
```bash
# Update system
sudo apt update && sudo apt full-upgrade -y

# Hailo runtime, firmware, TAPPAS, and DKMS kernel driver
sudo apt install -y hailo-all dkms

# GStreamer plugins and Python bindings
sudo apt install -y \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    python3-gst-1.0 \
    libgstreamer1.0-dev

# Reboot to load Hailo PCIe kernel driver
sudo reboot
```

#### Verify Hailo Hardware
```bash
hailortcli fw-control identify
# Expected: Board Name: Hailo-8, Firmware Version: 4.x.x
```

#### Python Environment
```bash
# CRITICAL: Use --system-site-packages for Hailo/GStreamer bindings
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install hailo-apps-infra (provides hailo Python module)
git clone https://github.com/hailo-ai/hailo-apps-infra.git
cd hailo-apps-infra && sudo ./install.sh && cd ..

# Install Python dependencies
pip install -r requirements.txt
```

#### Model Files
Place the following in the `models/` directory:
- `yolov8s_pose.hef` — Compiled YOLOv8s Pose model for Hailo-8
- `pose_lstm.pth` — Trained BiLSTM weights (optional; random init used if absent)

Download the HEF model:
```bash
# Option 1: Hailo Model Zoo CLI
hailomz compile yolov8s_pose --target hailo8 --output models/yolov8s_pose.hef

# Option 2: From Hailo Model Zoo repository
# https://github.com/hailo-ai/hailo_model_zoo
```

## Project Structure

```
rpi-theft-detector/
├── config.py               # Centralized constants (RTSP URL, model paths, tuning)
├── ingestion_engine.py     # GStreamer pipeline + Hailo metadata & video stream
├── thread_manager.py       # Lock-free deque bridge + consumer thread
├── action_classifier.py    # BiLSTM model + LOCF data cleaning
├── alert_handler.py        # File logging + webhook notifications
├── dashboard/              # Real-time Web Monitoring Dashboard
│   ├── bridge.py           # Thread-safe state container & MJPEG frame buffer
│   ├── metrics.py          # System metrics collector (CPU, RAM, temp, disk)
│   ├── server.py           # Flask server (SSE stream + MJPEG /api/video_feed)
│   └── static/
│       └── index.html      # Single-page Glassmorphism UI
├── main.py                 # Entrypoint — orchestrates pipeline & dashboard
├── verify_architecture.py  # 19 unit tests (runs without hardware)
├── requirements.txt        # pip dependencies
├── models/
│   ├── yolov8s_pose.hef    # Hailo NPU model (you provide)
│   └── pose_lstm.pth       # BiLSTM weights (optional)
└── README.md
```

## Configuration

All constants are in [`config.py`](config.py). Key settings:

| Variable | Default | Description |
|---|---|---|
| `RTSP_URI` | `rtsp://admin:ODPEDI@192.168.0.102:554/h264/ch1/sub/av_stream` | Camera RTSP URL |
| `HEF_MODEL_PATH` | `models/yolov8s_pose.hef` | Hailo pose model |
| `CONFIDENCE_THRESHOLD` | `0.45` | LOCF trigger threshold |
| `INFERENCE_INTERVAL_SEC` | `0.1` | BiLSTM polling rate (10 Hz) |
| `THEFT_CONFIDENCE_THRESHOLD` | `0.7` | Min probability for alert |
| `ALERT_COOLDOWN_SEC` | `30.0` | Seconds between alerts |
| `WEBHOOK_URL` | `""` (env var) | HTTP endpoint for alerts |

Override via environment variables:
```bash
export RTSP_URI="rtsp://user:pass@10.0.0.50:554/stream"
export WEBHOOK_URL="https://hooks.slack.com/services/..."
```

## Usage

### Run the Pipeline
```bash
source venv/bin/activate
python main.py
```

### Run Tests (No Hardware Required)
```bash
python verify_architecture.py
# or with pytest:
python -m pytest verify_architecture.py -v
```

### Enable GStreamer Debug Logging
```bash
GST_DEBUG=2 python main.py        # Warnings + errors
GST_DEBUG=hailonet:4 python main.py  # Verbose Hailo element logs
```

## GStreamer Pipeline Graph

```
rtspsrc location=rtsp://... latency=100 drop-on-latency=true protocols=tcp
  ! rtph264depay
  ! h264parse
  ! avdec_h264 (or v4l2h264dec)           ← Auto-detected (avdec_h264 on RPi5, v4l2h264dec on RPi4)
  ! video/x-raw,format=NV12
  ! videoconvert
  ! video/x-raw,format=RGB,width=640,height=480
  ! queue max-size-buffers=3 leaky=downstream
  ! hailonet hef-path=models/yolov8s_pose.hef batch-size=1
  ! queue max-size-buffers=3 leaky=downstream
  ! hailofilter so-path=/.../libyolov8pose_post.so qos=false
  ! queue max-size-buffers=3 leaky=downstream
  ! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `hailortcli` not found | `sudo apt install hailo-all && sudo reboot` |
| `import hailo` fails | Recreate venv with `--system-site-packages` |
| `no element "v4l2h264dec"` | RPi 5 uses software decoding (`avdec_h264`). Pipeline now auto-detects decoders. Ensure `gstreamer1.0-libav` is installed: `sudo apt install gstreamer1.0-libav` |
| RTSP connection timeout | Check camera IP, port, and credentials |
| High CPU usage | Increase `INFERENCE_INTERVAL_SEC` in config.py |
| Alert flood | Increase `ALERT_COOLDOWN_SEC` |

## License

MIT
