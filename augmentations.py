

from typing import Tuple
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


class ImageAugmentationPipeline:
    """Configurable data augmentation pipeline built with tf.keras.layers."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        random_flip: str = "horizontal",
        rotation_factor: float = 0.15,
        zoom_factor: float = 0.15,
        translation_factor: float = 0.1,
        brightness_factor: float = 0.1
    ):
        """Initializes the image augmentation pipeline.

        Args:
            image_size: Input target spatial resolution (height, width).
            random_flip: Flip direction ('horizontal', 'vertical', 'horizontal_and_vertical').
            rotation_factor: Maximum rotation angle fraction.
            zoom_factor: Range for random zooming.
            translation_factor: Fractional height and width shifts.
            brightness_factor: Random brightness change magnitude.
        """
        self.image_size = image_size
        self.augmentation_layers = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(random_flip),
                tf.keras.layers.RandomRotation(rotation_factor),
                tf.keras.layers.RandomZoom(zoom_factor),
                tf.keras.layers.RandomTranslation(translation_factor, translation_factor),
                tf.keras.layers.RandomBrightness(brightness_factor),
            ],
            name="data_augmentation"
        )
        logger.info("Initialized tf.keras data augmentation pipeline.")

    def get_augmentation_layer(self) -> tf.keras.layers.Layer:
        """Returns the sequential augmentation layer."""
        return self.augmentation_layers

    def __call__(self, images: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Applies data augmentation to a batch of images.

        Args:
            images: Tensor of batch shape (B, H, W, C).
            training: Whether to apply augmentations (true in training mode).

        Returns:
            tf.Tensor: Augmented image batch.
        """
        if training:
            return self.augmentation_layers(images, training=True)
        return images
