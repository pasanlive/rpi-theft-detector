from __future__ import annotations

import itertools
import logging
import threading
from collections import deque
from typing import Callable, Optional

import numpy as np

from config import SEQUENCE_LENGTH, FEATURE_DIM, INFERENCE_INTERVAL_SEC

logger = logging.getLogger(__name__)


class ThreadBridge:
    """
    A lock-free producer-consumer bridge for passing feature vectors 
    from the GStreamer/Hailo NPU pipeline to the PyTorch BiLSTM consumer.
    """

    def __init__(self) -> None:
        """
        Initializes the ThreadBridge with a fixed-size deque.
        """
        self._deque: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        # Using itertools.count to avoid manual locking for the counter in CPython
        self._counter = itertools.count()
        self._frame_count = 0

    def push(self, vector: np.ndarray) -> None:
        """
        Appends a feature vector to the bridge. 
        Expected to be a 51-dim float32 vector from the GStreamer appsink.
        
        Args:
            vector (np.ndarray): The feature vector to append.
        """
        self._deque.append(vector)
        self._frame_count = next(self._counter) + 1

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Retrieves the current sequence snapshot as a numpy array.
        
        Returns:
            Optional[np.ndarray]: A sequence array of shape [SEQUENCE_LENGTH, FEATURE_DIM] 
            if enough frames are available, else None.
        """
        if not self.is_ready():
            return None
        
        # list(deque) is thread-safe (GIL-atomic) in CPython
        seq_list = list(self._deque)
        
        # Double check length in case of a race condition emptying the deque
        if len(seq_list) < SEQUENCE_LENGTH:
            return None
            
        return np.array(seq_list, dtype=np.float32)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Retrieves the most recent feature vector pushed to the bridge.

        Returns:
            Optional[np.ndarray]: Vector of shape [FEATURE_DIM] if available, else None.
        """
        if len(self._deque) > 0:
            return self._deque[-1]
        return None

    def is_ready(self) -> bool:
        """
        Checks if the bridge has enough frames for inference.
        
        Returns:
            bool: True if length of the deque is at least SEQUENCE_LENGTH.
        """
        return len(self._deque) >= SEQUENCE_LENGTH

    def clear(self) -> None:
        """
        Empties the deque.
        """
        self._deque.clear()

    @property
    def frame_count(self) -> int:
        """
        Gets the total number of frames pushed to the bridge.
        
        Returns:
            int: The total frame count.
        """
        return self._frame_count


class ConsumerThread(threading.Thread):
    """
    A daemon thread that polls the ThreadBridge, runs the classifier, 
    and issues callbacks with the results.
    """

    def __init__(
        self, 
        bridge: ThreadBridge, 
        classifier_fn: Callable[[np.ndarray], tuple[str, float]], 
        result_callback: Callable[[str, float], None], 
        interval: float = INFERENCE_INTERVAL_SEC
    ) -> None:
        """
        Initializes the ConsumerThread.
        
        Args:
            bridge (ThreadBridge): The lock-free bridge connecting producer and consumer.
            classifier_fn (Callable): Function that takes an [N, D] array and returns (label, confidence).
            result_callback (Callable): Callback invoked with the result of classifier_fn.
            interval (float): The sleep interval between checks.
        """
        super().__init__(daemon=True)
        self.bridge = bridge
        self.classifier_fn = classifier_fn
        self.result_callback = result_callback
        self.interval = interval
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """
        Signals the thread to stop and cleanly exit.
        """
        self._stop_event.set()

    def run(self) -> None:
        """
        Main loop of the consumer thread. Periodically checks the bridge,
        runs inference if ready, and handles exceptions safely.
        """
        logger.info("ConsumerThread started. Polling interval: %.3fs", self.interval)
        
        while not self._stop_event.is_set():
            try:
                if self.bridge.is_ready():
                    sequence = self.bridge.get_sequence()
                    if sequence is not None:
                        label, confidence = self.classifier_fn(sequence)
                        self.result_callback(label, confidence)
            except Exception as e:
                logger.error("Exception in ConsumerThread loop: %s", e, exc_info=True)
            
            # Use wait instead of time.sleep so the thread can be stopped immediately
            self._stop_event.wait(self.interval)
        
        logger.info("ConsumerThread stopped.")
