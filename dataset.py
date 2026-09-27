

import os
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import cv2
import tensorflow as tf
from preprocessing.augmentations import ImageAugmentationPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class ProductDatasetPipeline:
    """ETL data pipeline converting image files into optimized tf.data.Dataset streams."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        normalize: bool = True
    ):
        """Initializes dataset loader pipeline.

        Args:
            image_size: Target image dimensions (height, width).
            batch_size: Batch size for training/inference.
            normalize: Scale pixel values to range [0, 1].
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.normalize = normalize

    def decode_and_preprocess_image(self, file_path: tf.Tensor) -> tf.Tensor:
        """Reads, decodes, resizes, and normalizes a single image file tensor.

        Args:
            file_path: String Tensor containing image path.

        Returns:
            tf.Tensor: Decoded float32 image tensor of shape (H, W, 3).
        """
        img_bytes = tf.io.read_file(file_path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, self.image_size)
        img = tf.cast(img, tf.float32)
        if self.normalize:
            img = img / 255.0
        return img

    def build_dataset_from_directory(
        self,
        directory: Union[str, Path],
        is_training: bool = True,
        augment_pipeline: Optional[ImageAugmentationPipeline] = None
    ) -> Tuple[tf.data.Dataset, List[str], List[int]]:
        """Scans image directory organized by sub-category folders and constructs tf.data stream.

        Args:
            directory: Root image directory path.
            is_training: If True, shuffles and applies augmentations.
            augment_pipeline: ImageAugmentationPipeline instance.

        Returns:
            Tuple containing:
                - tf.data.Dataset emitting (image_batch, label_batch)
                - List of image filenames/filepaths
                - List of integer class labels
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {directory_path}")

        image_paths = []
        labels = []
        class_names = sorted([d.name for d in directory_path.iterdir() if d.is_dir()])
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = directory_path / class_name
            for file_path in class_dir.glob("*"):
                if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    image_paths.append(str(file_path))
                    labels.append(class_to_idx[class_name])

        if not image_paths:
            logger.warning(f"No image files found in {directory_path}")
            return tf.data.Dataset.from_tensor_slices(([], [])), [], []

        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)

        image_ds = path_ds.map(
            self.decode_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = tf.data.Dataset.zip((image_ds, label_ds))

        if is_training:
            dataset = dataset.shuffle(buffer_size=min(len(image_paths), 1000))
            if augment_pipeline:
                aug_layer = augment_pipeline.get_augmentation_layer()
                dataset = dataset.map(
                    lambda x, y: (aug_layer(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )

        dataset = dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        logger.info(f"Loaded dataset with {len(image_paths)} images across {len(class_names)} classes.")
        return dataset, image_paths, labels

    def load_single_image(self, file_path_or_bytes: Union[str, Path, bytes]) -> tf.Tensor:
        """Decodes and resizes a single image for direct inference.

        Args:
            file_path_or_bytes: File path string/Path or raw byte stream.

        Returns:
            tf.Tensor: Single image batch tensor of shape (1, H, W, 3).
        """
        if isinstance(file_path_or_bytes, bytes):
            img_np = cv2.imdecode(np.frombuffer(file_path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_np is None:
                raise ValueError("Could not decode image from byte buffer.")
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_np = cv2.resize(img_np, (self.image_size[1], self.image_size[0]))
            img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)
        else:
            path_str = str(file_path_or_bytes)
            img_tensor = self.decode_and_preprocess_image(tf.constant(path_str))

        if self.normalize and tf.reduce_max(img_tensor) > 1.0:
            img_tensor = img_tensor / 255.0

        return tf.expand_dims(img_tensor, axis=0)
