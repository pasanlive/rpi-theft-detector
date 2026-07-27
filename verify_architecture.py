"""Phase 4: Autonomous Verification of the Pipeline Architecture.

Tests the PyTorch BiLSTM model, LOCF data cleaning, and thread bridge
without requiring actual hardware (no GStreamer, no Hailo NPU).

These tests run on any machine with Python, NumPy, and PyTorch.

Usage::

    python -m pytest verify_architecture.py -v
    # or directly:
    python verify_architecture.py
"""

from __future__ import annotations

import sys
import threading
import time
import unittest

import numpy as np
import torch

# Ensure the project root is on sys.path for config imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    NUM_CLASSES,
    CONFIDENCE_THRESHOLD,
    NUM_KEYPOINTS,
    KEYPOINT_DIM,
    LSTM_INPUT_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    ACTION_LABELS,
)
from action_classifier import PoseActionLSTM, apply_locf, ActionClassifier
from thread_manager import ThreadBridge


class TestLSTMOutputShape(unittest.TestCase):
    """Verify LSTM model produces correct output dimensions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize model once for all shape tests."""
        cls.model = PoseActionLSTM(
            input_dim=LSTM_INPUT_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=LSTM_DROPOUT,
        )
        cls.model.eval()

    def test_single_sequence_output_shape(self) -> None:
        """Generate [1, 30, 51] input → assert output is [1, 2].

        This is the core verification from the spec: a single batch of
        30 frames with 17 keypoints * 3 dimensions = 51 features must
        produce exactly 2 class logits.
        """
        dummy = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)

        with torch.no_grad():
            output = self.model(dummy)

        self.assertEqual(
            output.shape,
            (1, NUM_CLASSES),
            f"Expected output shape [1, {NUM_CLASSES}], got {list(output.shape)}",
        )

    def test_batch_output_shape(self) -> None:
        """Generate [4, 30, 51] input → assert output is [4, 2].

        Verifies the model handles batch processing correctly for
        potential future multi-person scenarios.
        """
        batch_size = 4
        dummy = torch.randn(batch_size, SEQUENCE_LENGTH, FEATURE_DIM)

        with torch.no_grad():
            output = self.model(dummy)

        self.assertEqual(
            output.shape,
            (batch_size, NUM_CLASSES),
            f"Expected output shape [{batch_size}, {NUM_CLASSES}], "
            f"got {list(output.shape)}",
        )

    def test_single_frame_sequence(self) -> None:
        """Verify model handles a sequence of length 1 (edge case)."""
        dummy = torch.randn(1, 1, FEATURE_DIM)

        with torch.no_grad():
            output = self.model(dummy)

        self.assertEqual(output.shape, (1, NUM_CLASSES))


class TestSoftmaxProbability(unittest.TestCase):
    """Verify softmax output forms a valid probability distribution."""

    def test_softmax_sums_to_one(self) -> None:
        """Output probabilities must sum to 1.0 (within float tolerance)."""
        model = PoseActionLSTM(
            input_dim=LSTM_INPUT_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=LSTM_DROPOUT,
        )
        model.eval()

        dummy = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)

        with torch.no_grad():
            logits = model(dummy)
            probs = torch.softmax(logits, dim=1)

        prob_sum = probs.sum(dim=1).item()
        self.assertAlmostEqual(
            prob_sum, 1.0, places=5,
            msg=f"Softmax probabilities sum to {prob_sum}, expected 1.0",
        )

    def test_probabilities_nonnegative(self) -> None:
        """All softmax probabilities must be >= 0."""
        model = PoseActionLSTM()
        model.eval()
        dummy = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)

        with torch.no_grad():
            probs = torch.softmax(model(dummy), dim=1)

        self.assertTrue(
            (probs >= 0).all().item(),
            "Softmax produced negative probabilities.",
        )


class TestMemoryAlignment(unittest.TestCase):
    """Verify tensor memory contiguity — critical on ARM64."""

    def test_input_tensor_contiguity(self) -> None:
        """Input tensor must be contiguous in memory."""
        dummy = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)
        self.assertTrue(
            dummy.is_contiguous(),
            "Input tensor is not contiguous in memory.",
        )

    def test_output_tensor_contiguity(self) -> None:
        """Output tensor from LSTM must be contiguous."""
        model = PoseActionLSTM()
        model.eval()
        dummy = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)

        with torch.no_grad():
            output = model(dummy)

        self.assertTrue(
            output.is_contiguous(),
            "Output tensor is not contiguous in memory.",
        )

    def test_numpy_to_torch_roundtrip(self) -> None:
        """Verify numpy → torch → numpy preserves contiguity and values."""
        np_array = np.random.randn(SEQUENCE_LENGTH, FEATURE_DIM).astype(
            np.float32
        )
        tensor = torch.from_numpy(np_array).unsqueeze(0)  # [1, 30, 51]

        self.assertTrue(tensor.is_contiguous())
        self.assertEqual(tensor.dtype, torch.float32)

        # Roundtrip back to numpy
        roundtrip = tensor.squeeze(0).numpy()
        np.testing.assert_array_almost_equal(np_array, roundtrip)


class TestLOCFImputation(unittest.TestCase):
    """Verify Last Observation Carried Forward data cleaning."""

    def test_locf_basic_imputation(self) -> None:
        """Low-confidence keypoints should be overwritten with last valid."""
        seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        # Keypoint 0: Set frame 0 as valid, frames 1-2 as low confidence
        kp = 0
        x_idx, y_idx, c_idx = kp * 3, kp * 3 + 1, kp * 3 + 2

        # Frame 0: valid observation at (0.5, 0.6) with confidence 0.9
        seq[0, x_idx] = 0.5
        seq[0, y_idx] = 0.6
        seq[0, c_idx] = 0.9

        # Frame 1: low confidence (should be overwritten)
        seq[1, x_idx] = 0.1  # garbage value
        seq[1, y_idx] = 0.2  # garbage value
        seq[1, c_idx] = 0.2  # below threshold

        # Frame 2: low confidence (should also be overwritten)
        seq[2, x_idx] = 0.9
        seq[2, y_idx] = 0.8
        seq[2, c_idx] = 0.1

        cleaned = apply_locf(seq, confidence_threshold=CONFIDENCE_THRESHOLD)

        # Frame 1 should now carry forward from frame 0
        self.assertAlmostEqual(cleaned[1, x_idx], 0.5)
        self.assertAlmostEqual(cleaned[1, y_idx], 0.6)
        self.assertAlmostEqual(cleaned[1, c_idx], CONFIDENCE_THRESHOLD)

        # Frame 2 should also carry forward from frame 0
        self.assertAlmostEqual(cleaned[2, x_idx], 0.5)
        self.assertAlmostEqual(cleaned[2, y_idx], 0.6)
        self.assertAlmostEqual(cleaned[2, c_idx], CONFIDENCE_THRESHOLD)

    def test_locf_all_invalid(self) -> None:
        """When ALL frames have low confidence, coordinates stay as-is.

        Edge case: no valid observation exists to carry forward.
        The original (possibly zero) values are preserved.
        """
        seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        kp = 5  # left_shoulder
        c_idx = kp * 3 + 2

        # Set all frames to low confidence
        seq[:, c_idx] = 0.1  # All below threshold

        cleaned = apply_locf(seq, confidence_threshold=CONFIDENCE_THRESHOLD)

        # X, Y should remain zeros (no valid observation to carry forward)
        np.testing.assert_array_equal(
            cleaned[:, kp * 3 : kp * 3 + 2],
            seq[:, kp * 3 : kp * 3 + 2],
        )

    def test_locf_preserves_valid_keypoints(self) -> None:
        """Valid keypoints (conf >= threshold) must not be modified."""
        seq = np.random.rand(SEQUENCE_LENGTH, FEATURE_DIM).astype(np.float32)

        # Set all confidences above threshold
        for k in range(NUM_KEYPOINTS):
            seq[:, k * 3 + 2] = 0.9

        cleaned = apply_locf(seq, confidence_threshold=CONFIDENCE_THRESHOLD)

        np.testing.assert_array_almost_equal(
            seq, cleaned,
            err_msg="LOCF modified valid keypoints — should be no-op.",
        )

    def test_locf_does_not_mutate_input(self) -> None:
        """apply_locf must return a copy, not mutate the input array."""
        seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        seq[0, 2] = 0.9  # frame 0, kp 0 conf = valid
        seq[1, 2] = 0.1  # frame 1, kp 0 conf = invalid

        original = seq.copy()
        _ = apply_locf(seq, confidence_threshold=CONFIDENCE_THRESHOLD)

        np.testing.assert_array_equal(
            seq, original,
            err_msg="apply_locf mutated the input array.",
        )

    def test_locf_forward_only(self) -> None:
        """LOCF should only carry FORWARD (past → future), never backward."""
        seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        kp = 0
        x_idx, y_idx, c_idx = kp * 3, kp * 3 + 1, kp * 3 + 2

        # Frame 0: INVALID (no previous valid observation)
        seq[0, x_idx] = 0.1
        seq[0, y_idx] = 0.2
        seq[0, c_idx] = 0.1  # below threshold

        # Frame 5: VALID
        seq[5, x_idx] = 0.8
        seq[5, y_idx] = 0.9
        seq[5, c_idx] = 0.95

        cleaned = apply_locf(seq, confidence_threshold=CONFIDENCE_THRESHOLD)

        # Frame 0 should NOT be overwritten (no previous valid frame)
        self.assertAlmostEqual(cleaned[0, x_idx], 0.1)
        self.assertAlmostEqual(cleaned[0, y_idx], 0.2)


class TestThreadBridgeSafety(unittest.TestCase):
    """Verify thread-safe deque operations under concurrent access."""

    def test_concurrent_push_read(self) -> None:
        """Multiple producer threads pushing, one consumer reading.

        This stress-tests the GIL-atomic assumption for deque operations.
        No crashes or data corruption should occur.
        """
        bridge = ThreadBridge()
        num_producers = 4
        pushes_per_producer = 500
        errors: list[Exception] = []

        def producer(tid: int) -> None:
            try:
                for i in range(pushes_per_producer):
                    vec = np.full(FEATURE_DIM, tid * 1000 + i, dtype=np.float32)
                    bridge.push(vec)
            except Exception as e:
                errors.append(e)

        def consumer() -> None:
            try:
                for _ in range(200):
                    _ = bridge.get_sequence()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=producer, args=(i,))
            for i in range(num_producers)
        ]
        threads.append(threading.Thread(target=consumer))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(
            len(errors), 0,
            f"Thread safety violation: {errors}",
        )

    def test_deque_maxlen_enforced(self) -> None:
        """Bridge deque must never exceed SEQUENCE_LENGTH elements."""
        bridge = ThreadBridge()

        # Push more than maxlen
        for i in range(SEQUENCE_LENGTH * 3):
            vec = np.ones(FEATURE_DIM, dtype=np.float32) * i
            bridge.push(vec)

        seq = bridge.get_sequence()
        self.assertIsNotNone(seq)
        self.assertEqual(seq.shape, (SEQUENCE_LENGTH, FEATURE_DIM))

    def test_get_sequence_returns_none_when_insufficient(self) -> None:
        """get_sequence() returns None before SEQUENCE_LENGTH frames."""
        bridge = ThreadBridge()

        # Push fewer than required
        for i in range(SEQUENCE_LENGTH - 1):
            bridge.push(np.zeros(FEATURE_DIM, dtype=np.float32))

        self.assertIsNone(bridge.get_sequence())

    def test_frame_count_increments(self) -> None:
        """frame_count property must track total pushes accurately."""
        bridge = ThreadBridge()
        for _ in range(100):
            bridge.push(np.zeros(FEATURE_DIM, dtype=np.float32))

        self.assertEqual(bridge.frame_count, 100)


class TestActionClassifierIntegration(unittest.TestCase):
    """End-to-end test of the ActionClassifier wrapper."""

    def test_predict_returns_valid_label_and_confidence(self) -> None:
        """predict() must return a label from ACTION_LABELS and a float."""
        classifier = ActionClassifier(model_path=None)

        sequence = np.random.rand(SEQUENCE_LENGTH, FEATURE_DIM).astype(
            np.float32
        )
        # Set all confidences high so LOCF is a no-op
        for k in range(NUM_KEYPOINTS):
            sequence[:, k * 3 + 2] = 0.9

        label, confidence = classifier.predict(sequence)

        self.assertIn(label, ACTION_LABELS)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_predict_with_occluded_keypoints(self) -> None:
        """predict() should handle partially occluded inputs gracefully."""
        classifier = ActionClassifier(model_path=None)

        sequence = np.random.rand(SEQUENCE_LENGTH, FEATURE_DIM).astype(
            np.float32
        )
        # Set half the keypoints to low confidence
        for k in range(0, NUM_KEYPOINTS, 2):
            sequence[:, k * 3 + 2] = 0.1

        # Should not raise any exception
        label, confidence = classifier.predict(sequence)
        self.assertIn(label, ACTION_LABELS)


# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  RPi5 Theft Detector — Architecture Verification Suite")
    print("=" * 70)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  NumPy version   : {np.__version__}")
    print(f"  Feature dim     : {FEATURE_DIM}")
    print(f"  Sequence length : {SEQUENCE_LENGTH}")
    print(f"  LSTM hidden     : {LSTM_HIDDEN_DIM}")
    print(f"  Num classes     : {NUM_CLASSES}")
    print("=" * 70)

    unittest.main(verbosity=2)
