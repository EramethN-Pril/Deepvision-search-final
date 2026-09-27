

import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple, Literal, Optional
import numpy as np
import faiss
from utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:

    def __init__(
        self,
        embedding_dim: int = 128,
        metric_type: Literal["cosine", "euclidean"] = "cosine",
        db_path: Optional[Path] = None,
        index_file_path: Optional[Path] = None
    ):
        """Initializes FAISS Vector Store.

        Args:
            embedding_dim: Embedding feature length (default 128).
            metric_type: Similarity distance metric ('cosine' or 'euclidean').
            db_path: Path to SQLite metadata database.
            index_file_path: Path to binary FAISS index file on disk.
        """
        self.embedding_dim = embedding_dim
        self.metric_type = metric_type.lower()
        self.db_path = Path(db_path) if db_path else Path("data/metadata.db")
        self.index_file_path = Path(index_file_path) if index_file_path else Path("data/index/faiss.index")

        # Initialize FAISS Index
        if self.metric_type == "cosine":
            # Inner product on unit L2 normalized vectors equals Cosine Similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric_type == "euclidean":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported metric_type: '{metric_type}'")

        self._init_metadata_db()

    def _init_metadata_db(self) -> None:
       
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS product_catalog (
                    id INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    category TEXT NOT NULL
                )
            """)
            conn.commit()

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:
       
        if len(embeddings) == 0:
            logger.warning("Empty embeddings array supplied to vector store.")
            return

        embeddings_f32 = embeddings.astype(np.float32)

        # Normalize if metric is cosine
        if self.metric_type == "cosine":
            norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings_f32 = embeddings_f32 / norms

        start_id = self.index.ntotal
        self.index.add(embeddings_f32)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for idx, item in enumerate(metadata):
                catalog_id = start_id + idx
                cursor.execute("""
                    INSERT OR REPLACE INTO product_catalog (id, filename, filepath, category)
                    VALUES (?, ?, ?, ?)
                """, (catalog_id, item["filename"], item["filepath"], item["category"]))
            conn.commit()

        logger.info(f"Added {len(embeddings)} items to FAISS index. Total index count: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Searches top_k most similar items for a given query vector.

        Args:
            query_embedding: 1D or 2D numpy vector array of shape (128,) or (1, 128).
            top_k: Number of nearest neighbor items to retrieve.

        Returns:
            List[Dict[str, Any]]: List of matching results containing filename, filepath,
                category, similarity percentage score, and distance score.
        """
        if self.index.ntotal == 0:
            logger.warning("Search executed on empty FAISS index.")
            return []

        query = query_embedding.astype(np.float32)
        if len(query.shape) == 1:
            query = np.expand_dims(query, axis=0)

        if self.metric_type == "cosine":
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

        top_k_actual = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, top_k_actual)

        results = []
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue

                cursor.execute("SELECT filename, filepath, category FROM product_catalog WHERE id = ?", (int(idx),))
                row = cursor.fetchone()
                if row:
                    filename, filepath, category = row

                    if self.metric_type == "cosine":
                        # Cosine similarity score [0, 100%]
                        similarity_pct = max(0.0, min(100.0, float(score) * 100.0))
                    else:
                        # Convert Euclidean distance into similarity percentage
                        similarity_pct = max(0.0, min(100.0, (1.0 / (1.0 + float(score))) * 100.0))

                    results.append({
                        "id": int(idx),
                        "filename": filename,
                        "filepath": filepath,
                        "category": category,
                        "similarity": round(similarity_pct, 2),
                        "raw_score": round(float(score), 4)
                    })

        return results

    def save_index(self) -> None:
        """Saves binary FAISS index to disk."""
        self.index_file_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file_path))
        logger.info(f"Saved FAISS index to {self.index_file_path}")

    def load_index(self) -> bool:
        """Loads binary FAISS index from disk if existing.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if self.index_file_path.exists():
            self.index = faiss.read_index(str(self.index_file_path))
            logger.info(f"Loaded FAISS index from {self.index_file_path}. Total vectors: {self.index.ntotal}")
            return True
        logger.warning(f"Index file {self.index_file_path} not found.")
        return False
