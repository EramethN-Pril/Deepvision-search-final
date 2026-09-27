"""
Custom Deep Learning Loss functions for embedding learning.
Implementations include:
- Categorical Cross Entropy Loss
- Contrastive Loss (Pairwise distance margin)
- Triplet Loss (Anchor-Positive-Negative distance margin)
"""

from typing import Union, Callable
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


class ContrastiveLoss(tf.keras.losses.Loss):
    """Contrastive loss for pairwise metric learning.

    L(y, d) = y * d^2 + (1 - y) * max(margin - d, 0)^2
    where y=1 for similar pairs, y=0 for dissimilar pairs.
    """

    def __init__(self, margin: float = 0.5, name: str = "contrastive_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculates contrastive loss between paired embeddings.

        Args:
            y_true: Label tensor (1 for positive pair, 0 for negative pair).
            y_pred: Euclidean distance tensor between pair embeddings.

        Returns:
            tf.Tensor: Scalar contrastive loss value.
        """
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(self.margin - y_pred, 0.0))
        return tf.reduce_mean(y_true * square_pred + (1.0 - y_true) * margin_square)


class TripletLoss(tf.keras.losses.Loss):
    """Triplet loss for metric embedding learning.

    L(A, P, N) = max(d(A, P) - d(A, N) + margin, 0)
    """

    def __init__(self, margin: float = 0.5, name: str = "triplet_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Computes triplet loss from stacked/concatenated embedding predictions.

        Expects y_pred to contain concatenated embeddings [Anchor, Positive, Negative]
        or computes batch-all / batch-hard semi-hard triplet distance.
        """
        # If y_pred is batched embeddings (B, D) and y_true are labels (B,):
        # We compute pairwise euclidean distance matrix and form semi-hard triplets.
        labels = tf.cast(y_true, tf.int32)
        embeddings = y_pred

        # Pairwise distance matrix (B, B)
        dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)
        square_norm = tf.linalg.diag_part(dot_product)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
        distances = tf.maximum(distances, 0.0)

        # Mask for valid anchor-positive pairs (same label)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        mask_anchor_positive = tf.cast(labels_equal, tf.float32) - tf.eye(tf.shape(labels)[0])

        # Mask for valid anchor-negative pairs (different label)
        mask_anchor_negative = 1.0 - tf.cast(labels_equal, tf.float32)

        # Distance anchor-positive and anchor-negative
        anchor_positive_dist = tf.expand_dims(distances, 2)
        anchor_negative_dist = tf.expand_dims(distances, 1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Mask invalid triplets
        mask = tf.expand_dims(mask_anchor_positive, 2) * tf.expand_dims(mask_anchor_negative, 1)
        triplet_loss = tf.maximum(triplet_loss, 0.0) * mask

        # Count positive triplets
        valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
        num_positive_triplets = tf.reduce_sum(valid_triplets)

        return tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)


def get_loss_function(loss_name: str, margin: float = 0.5, num_classes: int = 10) -> tf.keras.losses.Loss:
    """Factory function to retrieve selected loss instance.

    Args:
        loss_name: One of 'categorical_crossentropy', 'contrastive_loss', 'triplet_loss'.
        margin: Margin threshold for metric losses.
        num_classes: Number of product classification categories.

    Returns:
        tf.keras.losses.Loss: Configured loss instance.
    """
    loss_name_clean = loss_name.lower().strip()
    logger.info(f"Configuring loss function: {loss_name_clean}")

    if loss_name_clean == "categorical_crossentropy":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    elif loss_name_clean == "contrastive_loss":
        return ContrastiveLoss(margin=margin)
    elif loss_name_clean == "triplet_loss":
        return TripletLoss(margin=margin)
    else:
        raise ValueError(f"Unknown loss function specified: '{loss_name}'")
