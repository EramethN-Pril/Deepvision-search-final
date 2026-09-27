"""
Model weight checkpointing and artifact persistence utility.
"""

from pathlib import Path
from typing import Optional
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages saving, loading, and inspecting Keras model weights."""

    def __init__(self, checkpoint_dir: Path):
        """Initializes checkpoint manager with target directory.

        Args:
            checkpoint_dir: Directory where model weights are stored.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_weights(self, model: tf.keras.Model, filename: str) -> Path:
        """Saves model weights to file.

        Args:
            model: Keras model instance.
            filename: Target file name (e.g. 'best_model.h5' or 'weights.ckpt').

        Returns:
            Path: Full path to saved weights file.
        """
        save_path = self.checkpoint_dir / filename
        model.save_weights(str(save_path))
        logger.info(f"Successfully saved model weights to: {save_path}")
        return save_path

    def load_weights(self, model: tf.keras.Model, filename: str) -> tf.keras.Model:
        """Loads weights into model instance.

        Args:
            model: Target Keras model.
            filename: Weight filename inside checkpoint directory.

        Returns:
            tf.keras.Model: Model instance with loaded weights.
        """
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        model.load_weights(str(load_path))
        logger.info(f"Successfully loaded model weights from: {load_path}")
        return model

    def get_latest_checkpoint(self, pattern: str = "*.h5") -> Optional[Path]:
        """Finds the most recently modified checkpoint matching pattern."""
        files = list(self.checkpoint_dir.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)
