"""
Batch embedding generator for catalog images using the trained Keras model.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Union, Any
import numpy as np
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generates feature embeddings for image catalog datasets."""

    def __init__(self, model: tf.keras.Model, batch_size: int = 32):
        """Initializes generator with embedding model.

        Args:
            model: Trained Keras feature embedding model.
            batch_size: Batch size for inference inference.
        """
        self.model = model
        self.batch_size = batch_size

    def extract_single_embedding(self, image_tensor: tf.Tensor) -> np.ndarray:
        """Extracts 128-dimensional normalized embedding for a single image tensor.

        Args:
            image_tensor: Image tensor of shape (1, H, W, 3) or (H, W, 3).

        Returns:
            np.ndarray: 1D normalized float32 numpy vector of shape (128,).
        """
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        embeddings = self.model(image_tensor, training=False)
        embedding_np = embeddings.numpy()[0]

        # Ensure L2 normalization
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
        return embedding_np.astype(np.float32)

    def extract_batch_embeddings(
        self,
        dataset: tf.data.Dataset,
        image_paths: List[str]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extracts feature embeddings for an entire dataset.

        Args:
            dataset: Preprocessed tf.data.Dataset emitting image batches.
            image_paths: Ordered list of full file paths matching dataset items.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]:
                - Matrix of shape (N, embedding_dim) containing extracted vectors.
                - List of metadata dictionaries (filename, path, category, id).
        """
        logger.info(f"Starting batch embedding generation for {len(image_paths)} images...")
        embeddings_list = []

        for batch_images, _ in dataset:
            batch_embeddings = self.model(batch_images, training=False).numpy()
            embeddings_list.append(batch_embeddings)

        if not embeddings_list:
            logger.warning("No embeddings generated; empty dataset provided.")
            return np.empty((0, 128), dtype=np.float32), []

        all_embeddings = np.vstack(embeddings_list)

        # Normalize rows to unit length for Cosine Similarity
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_embeddings = (all_embeddings / norms).astype(np.float32)

        metadata = []
        for idx, path_str in enumerate(image_paths):
            p = Path(path_str)
            metadata.append({
                "id": idx,
                "filename": p.name,
                "filepath": str(p.resolve()),
                "category": p.parent.name
            })

        logger.info(f"Successfully generated embeddings matrix of shape: {all_embeddings.shape}")
        return all_embeddings, metadata
