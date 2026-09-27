"""
Transfer Learning model architecture leveraging EfficientNetB0 backbone.
"""

from typing import Tuple
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


def build_efficientnet_b0_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    embedding_dim: int = 128,
    freeze_backbone: bool = True,
    l2_normalize: bool = True
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Constructs EfficientNetB0 transfer learning embedding model.

    Args:
        input_shape: Image input dimensions (H, W, C).
        embedding_dim: Feature embedding length (default 128).
        freeze_backbone: Whether to freeze pre-trained backbone layers initially.
        l2_normalize: Apply unit length L2 normalization to output vector.

    Returns:
        Tuple[tf.keras.Model, tf.keras.Model]: Full model, and base backbone model reference.
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")

    # EfficientNet pre-trained backbone
    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            pooling=None
        )
        logger.info("Loaded EfficientNetB0 with ImageNet pre-trained weights.")
    except Exception as e:
        logger.warning(f"Could not load pre-trained weights ({e}). Falling back to random initialization.")
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=inputs,
            pooling=None
        )

    base_model.trainable = not freeze_backbone

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_embedding")(x)
    x = tf.keras.layers.Dense(256, activation="swish", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    embeddings = tf.keras.layers.Dense(embedding_dim, name="raw_embedding")(x)

    if l2_normalize:
        embeddings = tf.keras.layers.Lambda(
            lambda v: tf.math.l2_normalize(v, axis=1),
            name="l2_normalized_embedding"
        )(embeddings)

    model = tf.keras.Model(inputs=inputs, outputs=embeddings, name="EfficientNetB0_Embedding_Model")
    logger.info(f"Built EfficientNetB0 model (Backbone Frozen: {freeze_backbone}).")
    return model, base_model


def set_backbone_trainable(base_model: tf.keras.Model, unfreeze_from_layer: int = -20) -> None:
    """Unfreezes upper layers of the base backbone for fine-tuning.

    Args:
        base_model: Base backbone model instance.
        unfreeze_from_layer: Layer index from which to set trainable=True.
    """
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
    logger.info(f"Unfrozen backbone layers from index {unfreeze_from_layer} onwards for fine-tuning.")
