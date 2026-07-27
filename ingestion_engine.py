"""Zero-Copy GStreamer → HailoRT Ingestion Engine.

Constructs a GStreamer pipeline that:
  1. Receives an RTSP H.264 stream
  2. Hardware-decodes via v4l2h264dec (RPi5 VideoCore VII)
  3. Runs YOLOv8s-Pose inference on the Hailo-8 NPU
  4. Extracts ONLY pose metadata (17 COCO keypoints) — zero image copy

The appsink callback never touches the image buffer. It reads
Hailo ROI metadata attached to the GstBuffer and pushes a 51-dim
keypoint vector into the ThreadBridge for downstream BiLSTM consumption.

Architecture Constraint:
    This module MUST NOT block. The appsink callback runs on the
    GStreamer streaming thread. Any blocking operation will stall
    the pipeline, cause RTSP buffer overflow, and crash the stream.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

COCO_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                  # Shoulders
    (5, 7), (7, 9),                          # Left arm
    (6, 8), (8, 10),                         # Right arm
    (5, 11), (6, 12),                        # Torso
    (11, 12),                                # Hips
    (11, 13), (13, 15),                      # Left leg
    (12, 14), (14, 16),                      # Right leg
]

if TYPE_CHECKING:
    from thread_manager import ThreadBridge

logger = logging.getLogger(__name__)

# ─── GStreamer / Hailo Imports ─────────────────────────────────────────────────
# These are system packages (apt: hailo-all, python3-gst-1.0).
# They are NOT available on non-RPi development machines.

try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GLib, GstApp  # noqa: E402
    GST_AVAILABLE = True
except (ImportError, ValueError) as exc:
    logger.warning("GStreamer bindings not available: %s", exc)
    GST_AVAILABLE = False

try:
    import hailo  # type: ignore[import-untyped]
    HAILO_AVAILABLE = True
except ImportError:
    logger.warning(
        "Hailo Python bindings not found. "
        "Ensure venv was created with --system-site-packages."
    )
    HAILO_AVAILABLE = False

from config import (
    RTSP_URI,
    HEF_MODEL_PATH,
    HAILO_POST_SO,
    NUM_KEYPOINTS,
    KEYPOINT_DIM,
    FEATURE_DIM,
    DETECTION_CONFIDENCE_MIN,
    GST_QUEUE_MAX_BUFFERS,
    GST_RTSP_LATENCY_MS,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


class IngestionEngine:
    """Manages the GStreamer pipeline and Hailo metadata extraction.

    Parameters
    ----------
    bridge : ThreadBridge
        The lock-free deque bridge to push keypoint vectors into.
    rtsp_uri : str, optional
        RTSP camera URL. Defaults to ``config.RTSP_URI``.
    hef_path : str, optional
        Path to the compiled ``.hef`` model. Defaults to ``config.HEF_MODEL_PATH``.

    Raises
    ------
    RuntimeError
        If GStreamer or Hailo bindings are not available.
    """

    def __init__(
        self,
        bridge: ThreadBridge,
        rtsp_uri: str = RTSP_URI,
        hef_path: str = HEF_MODEL_PATH,
        dash_bridge: Optional[Any] = None,
    ) -> None:
        if not GST_AVAILABLE:
            raise RuntimeError(
                "GStreamer Python bindings (gi.repository.Gst) are not installed. "
                "Run: sudo apt install python3-gst-1.0"
            )
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                "Hailo Python bindings are not installed. "
                "Run: sudo apt install hailo-all"
            )

        self._bridge = bridge
        self._dash_bridge = dash_bridge
        self._rtsp_uri = rtsp_uri
        self._hef_path = hef_path
        self._pipeline: Optional[Gst.Pipeline] = None
        self._main_loop: Optional[GLib.MainLoop] = None
        self._frame_width = FRAME_WIDTH
        self._frame_height = FRAME_HEIGHT
        self._sample_counter = 0

        Gst.init(None)
        logger.info(
            "IngestionEngine initialized — RTSP=%s, HEF=%s, CV2=%s, dashboard=%s",
            self._rtsp_uri, self._hef_path,
            CV2_AVAILABLE, self._dash_bridge is not None,
        )

    def build_pipeline(self) -> None:
        """Construct and link the GStreamer pipeline.

        Pipeline graph::

            rtspsrc (drop-on-latency) →
            rtph264depay → h264parse →
            v4l2h264dec (HW decode) →
            videoconvert → RGB →
            queue (leaky) →
            hailonet (NPU inference) →
            queue (leaky) →
            hailofilter (post-process → attach metadata) →
            queue (leaky) →
            appsink (metadata-only, drop=true)

        Design Notes:
        - Three ``queue`` elements with ``leaky=downstream`` create backpressure
          relief points. Under CPU load, oldest frames are silently dropped.
        - ``appsink max-buffers=1 drop=true`` ensures only the freshest frame
          is processed, preventing metadata lag.
        - ``sync=false`` disables clock synchronization — process ASAP.
        """
        # Dynamically detect available H.264 decoder
        # Note: Raspberry Pi 5 uses software decoding via `avdec_h264` (Cortex-A76),
        # while Raspberry Pi 4 uses hardware decoding via `v4l2h264dec`.
        decoder_elem = "avdec_h264"
        for dec in ["v4l2h264dec", "avdec_h264", "openh264dec"]:
            if Gst.ElementFactory.find(dec) is not None:
                decoder_elem = dec
                break
        logger.info("Using H.264 decoder element: %s", decoder_elem)

        pipeline_str = (
            f"rtspsrc name=src location={self._rtsp_uri} "
            f"  latency={GST_RTSP_LATENCY_MS} "
            f"  drop-on-latency=true "
            f"  protocols=tcp "
            f"rtph264depay name=depay "
            f"! h264parse "
            f"! {decoder_elem} "
            f"! videoconvert "
            f"! video/x-raw,format=RGB "
            f"! queue max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
            f"! hailonet hef-path={self._hef_path} batch-size=1 "
            f"  scheduling-algorithm=0 force-writable=true "
            f"! queue max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
            f"! hailofilter so-path={HAILO_POST_SO} qos=false "
            f"! queue max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
            f"! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )

        logger.info("Building GStreamer pipeline:\n%s", pipeline_str)

        self._pipeline = Gst.parse_launch(pipeline_str)
        if self._pipeline is None:
            raise RuntimeError("Failed to parse GStreamer pipeline string.")

        # Connect dynamic pad-added signal for RTSP source
        rtspsrc_elem = self._pipeline.get_by_name("src")
        if rtspsrc_elem is not None:
            rtspsrc_elem.connect("pad-added", self._on_rtspsrc_pad_added)

        # Connect appsink callback
        appsink = self._pipeline.get_by_name("sink")
        if appsink is None:
            raise RuntimeError("Failed to find 'sink' element in pipeline.")
        appsink.connect("new-sample", self._on_new_sample)

        # Connect bus for error / EOS handling
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::eos", self._on_bus_eos)
        bus.connect("message::warning", self._on_bus_warning)

        logger.info("GStreamer pipeline built successfully.")

    def _on_rtspsrc_pad_added(self, rtspsrc: Gst.Element, pad: Gst.Pad) -> None:
        """Dynamically link RTSP video pad to depayloader.

        RTSP streams create pads dynamically for video and audio media tracks.
        This handler selectively links the video pad to the depayloader and
        ignores unhandled tracks (e.g. audio), preventing 'not-linked' errors.
        """
        if self._pipeline is None:
            return
        depay = self._pipeline.get_by_name("depay")
        if depay is None:
            return
        sink_pad = depay.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return

        caps = pad.query_caps(None)
        caps_str = caps.to_string().lower() if caps is not None else ""
        logger.info("RTSP stream pad added: %s", caps_str)

        # Ignore audio tracks
        if "audio" in caps_str:
            logger.info("Ignoring audio stream pad: %s", caps_str)
            return

        # Link video stream pad to depayloader
        try:
            pad.link(sink_pad)
            logger.info("Successfully linked RTSP video stream to depayloader.")
        except Exception as exc:
            logger.warning("Could not link RTSP pad (%s): %s", caps_str, exc)

    def start(self) -> None:
        """Start the pipeline and enter the GLib main loop.

        This method blocks the calling thread. It should be called from
        the main thread after all other components are initialized.

        Raises
        ------
        RuntimeError
            If the pipeline has not been built via ``build_pipeline()``.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set pipeline to PLAYING state.")

        logger.info("GStreamer pipeline PLAYING — entering main loop.")
        self._main_loop = GLib.MainLoop()

        try:
            self._main_loop.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully shut down the pipeline and main loop."""
        logger.info("Stopping GStreamer pipeline...")
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        if self._main_loop is not None and self._main_loop.is_running():
            self._main_loop.quit()
        logger.info("GStreamer pipeline stopped.")

    # ─── Appsink Callback ──────────────────────────────────────────────────

    def _on_new_sample(self, sink: GstApp.AppSink) -> Gst.FlowReturn:
        """Extract Hailo pose metadata from the GstBuffer.

        This callback runs on the GStreamer streaming thread. It MUST be
        non-blocking. It does NOT read the image buffer — only metadata.

        Flow:
            1. Pull the GstSample from appsink
            2. Get the GstBuffer from the sample
            3. Extract the Hailo ROI (Region of Interest) metadata
            4. Iterate over person detections
            5. For each detection, extract 17 COCO landmarks
            6. Pack into a 51-dim float32 vector [X0,Y0,C0, X1,Y1,C1, ...]
            7. Push the vector into the ThreadBridge

        Only the highest-confidence person detection is forwarded to
        avoid ambiguity in single-person theft scenarios.

        Parameters
        ----------
        sink : GstApp.AppSink
            The appsink element emitting the signal.

        Returns
        -------
        Gst.FlowReturn
            ``OK`` on success, ``ERROR`` on critical failure.
        """
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        if buffer is None:
            return Gst.FlowReturn.OK

        # Log first frame arrival to confirm pipeline data flow
        if self._sample_counter == 0:
            logger.info(">>> FIRST FRAME arrived at appsink — pipeline data flow confirmed!")
            logger.info("    dash_bridge=%s, CV2_AVAILABLE=%s", self._dash_bridge is not None, CV2_AVAILABLE)



        best_detection = None
        points = None
        bbox = None

        try:
            # Step 1: Extract Hailo ROI metadata
            roi = hailo.get_roi_from_buffer(buffer)
            if roi is not None:
                # Step 2: Get all person detections
                detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
                best_confidence = 0.0
                for det in detections:
                    label = det.get_label().lower() if det.get_label() else ""
                    has_landmarks = len(det.get_objects_typed(hailo.HAILO_LANDMARKS)) > 0
                    is_person_det = (label == "person" or det.get_class_id() == 0 or has_landmarks or not label)
                    
                    if (is_person_det
                            and det.get_confidence() >= DETECTION_CONFIDENCE_MIN
                            and det.get_confidence() > best_confidence):
                        best_detection = det
                        best_confidence = det.get_confidence()

                if best_detection is not None:
                    landmarks_list = best_detection.get_objects_typed(hailo.HAILO_LANDMARKS)
                    if landmarks_list:
                        pts_candidate = landmarks_list[0].get_points()
                        if len(pts_candidate) >= NUM_KEYPOINTS:
                            points = pts_candidate
                            bbox = best_detection.get_bbox()

                            # Pack 51-dim vector for BiLSTM
                            vector = np.empty(FEATURE_DIM, dtype=np.float32)
                            for i in range(NUM_KEYPOINTS):
                                pt = points[i]
                                abs_x = pt.x() * bbox.width() + bbox.xmin()
                                abs_y = pt.y() * bbox.height() + bbox.ymin()
                                vector[i * KEYPOINT_DIM] = abs_x
                                vector[i * KEYPOINT_DIM + 1] = abs_y
                                vector[i * KEYPOINT_DIM + 2] = pt.confidence()

                            self._bridge.push(vector)

        except Exception:
            logger.exception("Error extracting Hailo metadata in appsink callback")

        # Step 3: Send video frame to dashboard if enabled
        if self._dash_bridge is not None and CV2_AVAILABLE:
            try:
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    # Dynamically query dimensions from sample caps
                    curr_w, curr_h = self._frame_width, self._frame_height
                    try:
                        caps = sample.get_caps()
                        if caps and caps.get_size() > 0:
                            st = caps.get_structure(0)
                            res_w, w_val = st.get_int("width")
                            res_h, h_val = st.get_int("height")
                            if res_w and res_h and w_val > 0 and h_val > 0:
                                curr_w, curr_h = w_val, h_val
                    except Exception:
                        pass

                    # Convert raw RGB buffer bytes to numpy array
                    rgb = np.frombuffer(map_info.data, dtype=np.uint8).reshape(
                        (curr_h, curr_w, 3)
                    )
                    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    # Draw pose & bbox overlay if person detected
                    if bbox is not None and points is not None:
                        self._draw_pose_overlay(frame_bgr, bbox, points)

                    # Compress frame to JPEG and push to dashboard bridge
                    ret_val, jpeg_buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret_val:
                        self._dash_bridge.push_jpeg_frame(jpeg_buf.tobytes())
                        self._sample_counter += 1
                        if self._sample_counter == 1 or self._sample_counter % 50 == 0:
                            logger.info("RTSP camera frame #%d pushed to live video feed.", self._sample_counter)
                    buffer.unmap(map_info)
            except Exception as exc:
                logger.error("Dashboard frame encoding error: %s", exc, exc_info=True)

        return Gst.FlowReturn.OK

    def _draw_pose_overlay(self, frame: np.ndarray, bbox: Any, points: Any) -> None:
        """Draw bounding box and 17 COCO pose keypoints on OpenCV BGR frame."""
        h, w, _ = frame.shape

        # Bounding box
        xmin = int(bbox.xmin() * w)
        ymin = int(bbox.ymin() * h)
        xmax = int((bbox.xmin() + bbox.width()) * w)
        ymax = int((bbox.ymin() + bbox.height()) * h)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (248, 189, 56), 2)

        # Points
        pts = []
        for pt in points:
            px = int((pt.x() * bbox.width() + bbox.xmin()) * w)
            py = int((pt.y() * bbox.height() + bbox.ymin()) * h)
            pts.append((px, py, pt.confidence()))

        # Skeleton lines
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(pts) and p2 < len(pts):
                x1, y1, c1 = pts[p1]
                x2, y2, c2 = pts[p2]
                if c1 >= 0.35 and c2 >= 0.35:
                    cv2.line(frame, (x1, y1), (x2, y2), (250, 139, 167), 2)

        # Keypoint circles
        for px, py, c in pts:
            if c >= 0.35:
                cv2.circle(frame, (px, py), 4, (52, 211, 153), -1)

    # ─── Bus Message Handlers ──────────────────────────────────────────────

    def _on_bus_error(
        self, bus: Gst.Bus, message: Gst.Message
    ) -> None:
        """Handle GStreamer pipeline errors."""
        err, debug = message.parse_error()
        logger.error(
            "GStreamer ERROR from %s: %s\nDebug: %s",
            message.src.get_name(), err.message, debug,
        )
        self.stop()

    def _on_bus_eos(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle end-of-stream (RTSP disconnection)."""
        logger.warning("GStreamer EOS — RTSP stream ended.")
        self.stop()

    def _on_bus_warning(
        self, bus: Gst.Bus, message: Gst.Message
    ) -> None:
        """Handle GStreamer warnings (non-fatal)."""
        warn, debug = message.parse_warning()
        logger.warning(
            "GStreamer WARNING from %s: %s\nDebug: %s",
            message.src.get_name(), warn.message, debug,
        )
