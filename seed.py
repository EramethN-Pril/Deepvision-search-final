"""
Random seed management module for experimental reproducibility.
"""

import os
import random
import numpy as np
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Sets seed across python random, numpy, and tensorflow for reproducibility.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Global seed set to: {seed}")
