

from typing import Tuple
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


def build_cnn_scratch_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    embedding_dim: int = 128,
    dropout_rate: float = 0.3,
    l2_normalize: bool = True
) -> tf.keras.Model:
    
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu", name="relu3")(x)

    # Pooling & Dense Embedding Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    embeddings = tf.keras.layers.Dense(embedding_dim, name="raw_embedding")(x)

    if l2_normalize:
        embeddings = tf.keras.layers.Lambda(
            lambda v: tf.math.l2_normalize(v, axis=1),
            name="l2_normalized_embedding"
        )(embeddings)

    model = tf.keras.Model(inputs=inputs, outputs=embeddings, name="CNN_Scratch_Embedding_Model")
    logger.info("Successfully constructed CNN Scratch model.")
    return model
