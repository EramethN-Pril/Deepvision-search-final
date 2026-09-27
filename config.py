
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Tuple, List, Literal


@dataclass
class PathConfig:
    """Directory and file path configurations."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    weights_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs")
    index_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "index")
    db_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "metadata.db")

    def create_directories(self) -> None:
        """Create required directories if they do not exist."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.weights_dir,
            self.logs_dir,
            self.index_dir,
            self.db_path.parent
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Deep learning model hyperparameters and configuration."""

    model_type: Literal["cnn_scratch", "efficientnet_b0"] = "efficientnet_b0"
    image_size: Tuple[int, int] = (224, 224)
    image_channels: int = 3
    embedding_dim: int = 128
    num_classes: int = 10
    dropout_rate: float = 0.3
    l2_normalize_embeddings: bool = True
    freeze_backbone: bool = True


@dataclass
class TrainingConfig:
    """Model training hyperparameters and optimization settings."""

    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    loss_function: Literal["categorical_crossentropy", "contrastive_loss", "triplet_loss"] = "triplet_loss"
    margin: float = 0.5  # For Contrastive and Triplet Loss
    patience_early_stopping: int = 5
    patience_reduce_lr: int = 3
    mixed_precision: bool = True
    seed: int = 42


@dataclass
class IndexConfig:
    """FAISS vector search indexing settings."""

    metric_type: Literal["cosine", "euclidean"] = "cosine"
    index_file_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "index" / "faiss.index")
    top_k: int = 5


@dataclass
class APIConfig:
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@dataclass
class Config:
    """Master configuration class bundling sub-configurations."""

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def __post_init__(self) -> None:
        """Initialize directory paths post object creation."""
        self.paths.create_directories()


# Global default configuration instance
config = Config()
