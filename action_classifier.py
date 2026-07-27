import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import (
    LSTM_INPUT_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    NUM_CLASSES, ACTION_LABELS, MODEL_WEIGHTS_PATH,
    CONFIDENCE_THRESHOLD, SEQUENCE_LENGTH, FEATURE_DIM,
)

logger = logging.getLogger(__name__)


class PoseActionLSTM(nn.Module):
    """
    BiLSTM neural network for classifying human actions from pose keypoint sequences.
    """
    def __init__(
        self,
        input_dim: int = 51,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),   # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, input_dim] = [B, 30, 51].

        Returns:
            torch.Tensor: Output logits of shape [B, num_classes].
        """
        # x shape: [batch, seq_len, input_dim] = [B, 30, 51]
        _, (h_n, _) = self.lstm(x)
        # h_n shape: [num_layers*2, B, hidden_dim]
        # Concatenate final forward (h_n[-2]) and backward (h_n[-1]) hidden states
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [B, hidden_dim*2]
        return self.classifier(h_cat)  # [B, num_classes]


def apply_locf(sequence: np.ndarray, confidence_threshold: float = 0.45) -> np.ndarray:
    """
    Applies Last Observation Carried Forward (LOCF) to missing/low-confidence keypoints.

    Args:
        sequence (np.ndarray): Shape [seq_len, 51] where 51 is 17 keypoints * 3 (X, Y, Conf).
        confidence_threshold (float): Minimum confidence for a keypoint to be considered valid.

    Returns:
        np.ndarray: Cleaned sequence (copy of original).
    """
    cleaned = sequence.copy()
    seq_len = cleaned.shape[0]
    num_keypoints = cleaned.shape[1] // 3

    for k in range(num_keypoints):
        x_idx = k * 3
        y_idx = k * 3 + 1
        conf_idx = k * 3 + 2

        # Boolean mask of valid frames for this keypoint
        valid = cleaned[:, conf_idx] >= confidence_threshold

        if not valid.any() or valid.all():
            continue

        # Get indices of valid and all frames
        valid_idx = np.where(valid)[0]
        all_idx = np.arange(seq_len)

        # For each frame, find the index of the previous valid frame in `valid_idx`
        # searchsorted with side='right' gives the index of the next valid frame.
        # Subtracting 1 gives the index of the previous valid frame.
        prev_idx = np.searchsorted(valid_idx, all_idx, side='right') - 1

        # A valid previous frame exists if prev_idx >= 0
        has_prev = prev_idx >= 0

        # We only want to fill frames that are invalid and have a valid previous frame
        invalid_has_prev = (~valid) & has_prev

        # Extract the actual frame indices of the previous valid observations
        prev_valid_frames = valid_idx[prev_idx[invalid_has_prev]]

        # Apply LOCF
        cleaned[invalid_has_prev, x_idx] = cleaned[prev_valid_frames, x_idx]
        cleaned[invalid_has_prev, y_idx] = cleaned[prev_valid_frames, y_idx]
        cleaned[invalid_has_prev, conf_idx] = confidence_threshold

    return cleaned


class ActionClassifier:
    """
    High-level wrapper for loading the LSTM model and performing inference.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = PoseActionLSTM(
            input_dim=LSTM_INPUT_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=LSTM_DROPOUT
        )

        path_to_load = model_path if model_path is not None else MODEL_WEIGHTS_PATH

        if path_to_load:
            try:
                self.model.load_state_dict(torch.load(path_to_load, map_location="cpu"))
                logger.info(f"Loaded model weights from {path_to_load}")
            except Exception as e:
                logger.warning(f"Failed to load weights from {path_to_load}, using random init: {e}")
        else:
            logger.warning("No model path provided, using random init.")

        self.model.eval()

    @torch.inference_mode()
    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Runs LOCF cleaning and inference on a single sequence.

        Args:
            sequence (np.ndarray): Shape [30, 51] for a sequence of 30 frames.

        Returns:
            Tuple[str, float]: Action label and confidence score.
        """
        # 1. Copy and apply LOCF cleaning
        cleaned = apply_locf(sequence, confidence_threshold=CONFIDENCE_THRESHOLD)

        # 2. Convert to torch tensor [1, 30, 51], dtype float32
        tensor = torch.tensor(cleaned, dtype=torch.float32).unsqueeze(0)

        # 3. Assert tensor is contiguous
        tensor = tensor.contiguous()
        assert tensor.is_contiguous(), "Tensor must be contiguous"

        # 4. Run through LSTM
        logits = self.model(tensor)

        # 5. Apply softmax
        probs = torch.softmax(logits, dim=1).squeeze(0)

        # 6. Return (action_label, confidence)
        max_prob, max_idx = torch.max(probs, dim=0)
        action_label = ACTION_LABELS[max_idx.item()]

        return action_label, max_prob.item()
