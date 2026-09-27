

from pathlib import Path
from typing import Optional
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    

    def __init__(self, checkpoint_dir: Path):
      
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_weights(self, model: tf.keras.Model, filename: str) -> Path:
      
        save_path = self.checkpoint_dir / filename
        model.save_weights(str(save_path))
        logger.info(f"Successfully saved model weights to: {save_path}")
        return save_path

    def load_weights(self, model: tf.keras.Model, filename: str) -> tf.keras.Model:
        
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        model.load_weights(str(load_path))
        logger.info(f"Successfully loaded model weights from: {load_path}")
        return model

    def get_latest_checkpoint(self, pattern: str = "*.h5") -> Optional[Path]:
     
        files = list(self.checkpoint_dir.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)
