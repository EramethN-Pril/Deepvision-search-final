"""
Unit tests for DeepVision Search data pipelines, model outputs, and FAISS indexing.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import ModelConfig, TrainingConfig
from models.factory import ModelFactory
from preprocessing.dataset import ProductDatasetPipeline
from embeddings.generator import EmbeddingGenerator
from embeddings.vector_store import FAISSVectorStore
from training.losses import ContrastiveLoss, TripletLoss
from data.synthetic_generator import create_synthetic_catalog


@pytest.fixture
def temp_catalog_dir(tmp_path):
    """Creates a temporary synthetic catalog directory."""
    catalog_dir = tmp_path / "catalog"
    create_synthetic_catalog(catalog_dir, num_samples_per_category=3)
    return catalog_dir


def test_cnn_scratch_model_output_shape():
    """Verifies scratch CNN outputs (B, 128) embedding matrix."""
    model_cfg = ModelConfig(model_type="cnn_scratch", embedding_dim=128)
    model, _ = ModelFactory.create_model(model_cfg)

    dummy_input = tf.random.uniform((2, 224, 224, 3))
    output = model(dummy_input)

    assert output.shape == (2, 128)
    # Verify L2 normalization
    norms = tf.norm(output, axis=1).numpy()
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_efficientnet_b0_model_output_shape():
    """Verifies EfficientNetB0 outputs (B, 128) embedding matrix."""
    model_cfg = ModelConfig(model_type="efficientnet_b0", embedding_dim=128)
    model, base_model = ModelFactory.create_model(model_cfg)

    dummy_input = tf.random.uniform((1, 224, 224, 3))
    output = model(dummy_input)

    assert output.shape == (1, 128)
    assert base_model is not None


def test_dataset_pipeline(temp_catalog_dir):
    """Tests loading and decoding images via ProductDatasetPipeline."""
    pipeline = ProductDatasetPipeline(image_size=(224, 224), batch_size=4)
    dataset, image_paths, labels = pipeline.build_dataset_from_directory(temp_catalog_dir, is_training=False)

    assert len(image_paths) == 15
    assert len(labels) == 15

    for images, batch_labels in dataset.take(1):
        assert images.shape == (4, 224, 224, 3)
        assert batch_labels.shape == (4,)


def test_faiss_vector_store_search(tmp_path):
    """Tests FAISS indexing and top-K similarity search logic."""
    db_path = tmp_path / "test_meta.db"
    index_path = tmp_path / "test.index"

    store = FAISSVectorStore(
        embedding_dim=128,
        metric_type="cosine",
        db_path=db_path,
        index_file_path=index_path
    )

    # Generate dummy embeddings
    embeddings = np.random.randn(10, 128).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    metadata = [
        {"filename": f"img_{i}.jpg", "filepath": f"/tmp/img_{i}.jpg", "category": "shoes"}
        for i in range(10)
    ]

    store.add_embeddings(embeddings, metadata)
    assert store.index.ntotal == 10

    # Query with exact first vector
    results = store.search(embeddings[0], top_k=3)
    assert len(results) == 3
    assert results[0]["id"] == 0
    assert results[0]["similarity"] > 99.0  # Exact match should be ~100%


def test_custom_losses():
    """Tests Contrastive and Triplet loss calculations."""
    c_loss_fn = ContrastiveLoss(margin=0.5)
    t_loss_fn = TripletLoss(margin=0.5)

    y_true = tf.constant([1.0, 0.0])
    y_pred = tf.constant([0.1, 0.9])
    c_val = c_loss_fn(y_true, y_pred)
    assert c_val.numpy() > 0.0

    embeddings = tf.random.normal((6, 128))
    labels = tf.constant([0, 0, 1, 1, 2, 2])
    t_val = t_loss_fn(labels, embeddings)
    assert not np.isnan(t_val.numpy())
