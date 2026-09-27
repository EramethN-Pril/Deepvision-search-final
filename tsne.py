"""
t-SNE embedding space visualization module.
Projects 128-dimensional product embeddings into 2D scatter plots color-coded by category.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from utils.logger import get_logger

logger = get_logger(__name__)


def visualize_tsne_embeddings(
    embeddings: np.ndarray,
    labels: List[str],
    output_path: Optional[Path] = None,
    perplexity: int = 15,
    n_iter: int = 1000
) -> plt.Figure:
    """Performs t-SNE reduction on embeddings and generates scatter plot visualization.

    Args:
        embeddings: Float32 matrix of shape (N, 128).
        labels: List of category string labels corresponding to rows.
        output_path: Optional output path to save PNG file.
        perplexity: t-SNE perplexity parameter.
        n_iter: Max optimization iterations.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    logger.info(f"Computing t-SNE projection for {len(embeddings)} high-dimensional vectors...")

    # Adjust perplexity if sample size is small
    n_samples = len(embeddings)
    actual_perplexity = min(perplexity, max(1, n_samples - 1))

    tsne = TSNE(n_components=2, perplexity=actual_perplexity, max_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    unique_labels = sorted(list(set(labels)))
    palette = sns.color_palette("husl", len(unique_labels))

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, category in enumerate(unique_labels):
        mask = [lbl == category for lbl in labels]
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[palette[idx]],
            label=category,
            alpha=0.8,
            edgecolors="k",
            s=60
        )

    ax.set_title("t-SNE Visualization of Visual Product Embedding Space")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(title="Product Categories", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        logger.info(f"Saved t-SNE embedding plot to {output_path}")

    return fig
