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
    CAMERA_SOURCE,
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
    camera_source : str, optional
        Camera source type: ``"picam"`` (Pi Camera 2), ``"rtsp"``, or ``"v4l2"``.
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
        camera_source: str = CAMERA_SOURCE,
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
        self._camera_source = camera_source.lower()
        self._rtsp_uri = rtsp_uri
        self._hef_path = hef_path
        self._pipeline: Optional[Gst.Pipeline] = None
        self._main_loop: Optional[GLib.MainLoop] = None
        self._frame_width = FRAME_WIDTH
        self._frame_height = FRAME_HEIGHT
        self._sample_counter = 0

        Gst.init(None)
        logger.info(
            "IngestionEngine initialized — source=%s, RTSP=%s, HEF=%s, CV2=%s, dashboard=%s",
            self._camera_source, self._rtsp_uri, self._hef_path,
            CV2_AVAILABLE, self._dash_bridge is not None,
        )

    def build_pipeline(self) -> None:
        """Construct and link the GStreamer pipeline for Pi Cam 2, RTSP, or V4L2."""
        if self._camera_source in ["picam", "libcamerasrc", "camera"]:
            if Gst.ElementFactory.find("libcamerasrc") is not None:
                logger.info("Configuring pipeline for Raspberry Pi Camera Module 2 (libcamerasrc)...")
                src_elem = "libcamerasrc name=src ! video/x-raw, width=1280, height=720, format=NV12"
            elif Gst.ElementFactory.find("v4l2src") is not None:
                logger.info("libcamerasrc not found — falling back to v4l2src (/dev/video0) for Pi Camera...")
                src_elem = "v4l2src name=src device=/dev/video0"
            else:
                logger.info("libcamerasrc and v4l2src not found — using autovideosrc...")
                src_elem = "autovideosrc name=src"

            pipeline_str = (
                f"{src_elem} "
                "! videoconvert "
                "! videoscale "
                "! video/x-raw, width=640, height=640, format=RGB "
                f"! queue max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
                f"! hailonet hef-path={self._hef_path} batch-size=1 "
                f"  scheduling-algorithm=0 force-writable=true "
                f"! queue max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
                f"! hailofilter so-path={HAILO_POST_SO} qos=false "
                f"! queue max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
                f"! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
            )
        elif self._camera_source in ["v4l2", "usb", "webcam"]:
            logger.info("Configuring pipeline for V4L2 USB Webcam (v4l2src)...")
            pipeline_str = (
                "v4l2src name=src device=/dev/video0 "
                "! videoconvert "
                "! videoscale "
                "! video/x-raw, width=640, height=640, format=RGB "
                f"! queue max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
                f"! hailonet hef-path={self._hef_path} batch-size=1 "
                f"  scheduling-algorithm=0 force-writable=true "
                f"! queue max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
                f"! hailofilter so-path={HAILO_POST_SO} qos=false "
                f"! queue max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
                f"! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
            )
        else:
            # Default to RTSP network stream
            decoder_elem = self._find_decoder("h265")
            logger.info("Configuring RTSP stream pipeline (using decoder: %s)...", decoder_elem)
            pipeline_str = (
                f'rtspsrc name=src location="{self._rtsp_uri}" '
                f"  latency=200 "
                f"  drop-on-latency=true "
                f"  protocols=tcp "
                f"! queue name=video_in max-size-buffers={GST_QUEUE_MAX_BUFFERS} leaky=downstream "
                f"! rtph265depay "
                f"! h265parse config-interval=-1 "
                f"! {decoder_elem} "
                f"! videoconvert "
                f"! videoscale "
                f"! video/x-raw, width=640, height=640, format=RGB "
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

        # Connect dynamic pad-added handler if using RTSP source
        rtspsrc_elem = self._pipeline.get_by_name("src")
        if rtspsrc_elem is not None and self._camera_source not in ["picam", "libcamerasrc", "v4l2", "usb"]:
            rtspsrc_elem.connect("pad-added", self._on_rtspsrc_pad_added)
            rtspsrc_elem.connect("no-more-pads", self._on_rtspsrc_no_more_pads)
            logger.info("Connected pad-added signal handler to rtspsrc element '%s'.", rtspsrc_elem.get_name())
        else:
            logger.error("CRITICAL: Could not find 'src' rtspsrc element in pipeline!")

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

    def _find_decoder(self, codec: str) -> str:
        """Find the best available decoder element for the given codec.

        Parameters
        ----------
        codec : str
            Either ``"h264"`` or ``"h265"``.

        Returns
        -------
        str
            The GStreamer element factory name for the decoder.
        """
        if codec == "h265":
            candidates = ["v4l2h265dec", "avdec_h265"]
        else:
            candidates = ["v4l2h264dec", "avdec_h264", "openh264dec"]

        for dec_name in candidates:
            if Gst.ElementFactory.find(dec_name) is not None:
                return dec_name

        # Last resort fallback
        return f"avdec_{codec}"

    def _on_rtspsrc_no_more_pads(self, rtspsrc: Gst.Element) -> None:
        """Log when rtspsrc has finished creating all dynamic pads."""
        logger.info("rtspsrc: no-more-pads — all RTSP stream pads have been created.")

    def _on_rtspsrc_pad_added(self, rtspsrc: Gst.Element, pad: Gst.Pad) -> None:
        """Route dynamic RTSP pads: video → video_in queue, audio → fakesink."""
        caps = pad.query_caps(None)
        caps_str = caps.to_string().lower() if caps is not None else ""
        logger.info("rtspsrc pad-added: pad=%s caps=%s", pad.get_name(), caps_str[:200])

        if self._pipeline is None:
            return

        # Route audio pads to fakesink to prevent 'not-linked' pipeline crash
        if "audio" in caps_str:
            logger.info("Routing audio pad '%s' to fakesink.", pad.get_name())
            fakesink = Gst.ElementFactory.make("fakesink", None)
            if fakesink is not None:
                fakesink.set_property("async", False)
                fakesink.set_property("sync", False)
                self._pipeline.add(fakesink)
                fakesink.sync_state_with_parent()
                sink_pad = fakesink.get_static_pad("sink")
                if sink_pad:
                    pad.link(sink_pad)
            return

        # Route video pads to video_in entry queue
        video_in = self._pipeline.get_by_name("video_in")
        if video_in is None:
            logger.error("Could not find 'video_in' queue in pipeline!")
            return

        sink_pad = video_in.get_static_pad("sink")
        if sink_pad is None:
            logger.error("Could not get sink pad of 'video_in' queue!")
            return

        if sink_pad.is_linked():
            logger.info("video_in sink pad is already linked, ignoring pad '%s'.", pad.get_name())
            return

        res = pad.link(sink_pad)
        logger.info("Linked RTSP video pad '%s' → video_in queue (result: %s)", pad.get_name(), res)

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
                    curr_w, curr_h = self._frame_width, self._frame_height
                    try:
                        caps = sample.get_caps()
                        if caps and caps.get_size() > 0:
                            st = caps.get_structure(0)
                            res_w, val_w = st.get_int("width")
                            res_h, val_h = st.get_int("height")
                            if res_w and val_w > 0:
                                curr_w = val_w
                            if res_h and val_h > 0:
                                curr_h = val_h
                    except Exception:
                        pass

                    actual_len = len(map_info.data)
                    expected_len = curr_h * curr_w * 3

                    # Auto-derive dimensions if mismatch detected (e.g. 640x640 NPU buffer)
                    if actual_len != expected_len:
                        if actual_len == 640 * 640 * 3:
                            curr_w, curr_h = 640, 640
                            expected_len = actual_len
                        elif actual_len == 640 * 480 * 3:
                            curr_w, curr_h = 640, 480
                            expected_len = actual_len
                        elif actual_len % (curr_w * 3) == 0:
                            curr_h = actual_len // (curr_w * 3)
                            expected_len = actual_len

                    if actual_len != expected_len:
                        logger.warning(
                            "Buffer length mismatch: expected %d (%dx%dx3), got %d bytes",
                            expected_len, curr_w, curr_h, actual_len
                        )
                    else:
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
