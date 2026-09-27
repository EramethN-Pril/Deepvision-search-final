"""
Evaluation pipeline computing Precision, Recall, F1, Top-K accuracy, Confusion Matrix, and ROC curves.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_curve, auc
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Computes quantitative metrics and visual plots for model evaluation."""

    def __init__(self, output_dir: Path):
        """Initializes evaluator with plot export path.

        Args:
            output_dir: Directory where evaluation plots are saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_retrieval_metrics(
        self,
        query_embeddings: np.ndarray,
        query_labels: np.ndarray,
        catalog_embeddings: np.ndarray,
        catalog_labels: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, float]:
        """Calculates Top-1 and Top-K retrieval precision/recall and accuracy.

        Args:
            query_embeddings: Query vectors matrix (N_q, D).
            query_labels: Ground truth class integers (N_q,).
            catalog_embeddings: Gallery catalog vectors matrix (N_c, D).
            catalog_labels: Gallery catalog class labels (N_c,).
            top_k: Top K nearest neighbor cut-off.

        Returns:
            Dict[str, float]: Dictionary of evaluated metrics.
        """
        top1_hits = 0
        topk_hits = 0
        total_queries = len(query_labels)

        # Compute cosine similarity matrix (N_q, N_c)
        sim_matrix = np.dot(query_embeddings, catalog_embeddings.T)

        for idx in range(total_queries):
            true_label = query_labels[idx]
            # Rank gallery items in descending order of similarity
            ranked_indices = np.argsort(-sim_matrix[idx])
            top_retrieved_labels = catalog_labels[ranked_indices[:top_k]]

            if top_retrieved_labels[0] == true_label:
                top1_hits += 1

            if true_label in top_retrieved_labels:
                topk_hits += 1

        top1_acc = float(top1_hits) / total_queries if total_queries > 0 else 0.0
        topk_acc = float(topk_hits) / total_queries if total_queries > 0 else 0.0

        metrics = {
            "top1_accuracy": round(top1_acc * 100.0, 2),
            f"top{top_k}_accuracy": round(topk_acc * 100.0, 2)
        }

        logger.info(f"Retrieval Evaluation Results: {metrics}")
        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        filename: str = "confusion_matrix.png"
    ) -> Path:
        """Plots and saves multi-class confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title("Visual Search Category Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {save_path}")
        return save_path

    def plot_training_history(self, history: Dict[str, Any], filename: str = "training_curves.png") -> Path:
        """Plots training and validation loss curves."""
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.get("loss", []), label="Train Loss")
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="Val Loss")
        plt.title("Model Loss Progression")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        if "accuracy" in history:
            plt.subplot(1, 2, 2)
            plt.plot(history["accuracy"], label="Train Accuracy")
            if "val_accuracy" in history:
                plt.plot(history["val_accuracy"], label="Val Accuracy")
            plt.title("Model Accuracy Progression")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved training history curves to {save_path}")
        return save_path
