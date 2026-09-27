

from typing import Tuple, Optional
import tensorflow as tf
from models.cnn_scratch import build_cnn_scratch_model
from models.transfer_learning import build_efficientnet_b0_model
from config.config import ModelConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory class to create Keras model instances based on configuration."""

    @staticmethod
    def create_model(model_config: ModelConfig) -> Tuple[tf.keras.Model, Optional[tf.keras.Model]]:
       
        Args:
            model_config: ModelConfig object specifying model_type, image_size, etc.

        Returns:
            Tuple[tf.keras.Model, Optional[tf.keras.Model]]:
                (Embedding Model, Base Backbone Model if transfer learning else None)
        """
        input_shape = (model_config.image_size[0], model_config.image_size[1], model_config.image_channels)
        logger.info(f"Instantiating model type: {model_config.model_type}")

        if model_config.model_type == "cnn_scratch":
            model = build_cnn_scratch_model(
                input_shape=input_shape,
                embedding_dim=model_config.embedding_dim,
                dropout_rate=model_config.dropout_rate,
                l2_normalize=model_config.l2_normalize_embeddings
            )
            return model, None

        elif model_config.model_type == "efficientnet_b0":
            model, base_model = build_efficientnet_b0_model(
                input_shape=input_shape,
                embedding_dim=model_config.embedding_dim,
                freeze_backbone=model_config.freeze_backbone,
                l2_normalize=model_config.l2_normalize_embeddings
            )
            return model, base_model

        else:
            raise ValueError(f"Unsupported model type requested: '{model_config.model_type}'")
