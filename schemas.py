

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):


    filename: str = Field(..., description="Image filename of matched product", example="shoe001.jpg")
    similarity: float = Field(..., description="Visual similarity score percentage", example=96.2)
    category: str = Field(..., description="Catalog category name", example="shoes")
    filepath: str = Field(..., description="Full disk path to catalog image file")
    raw_score: float = Field(..., description="Raw distance/similarity metric score")


class SearchResponse(BaseModel):
    """Response schema for image similarity search query."""

    inference_time_ms: float = Field(..., description="Model inference & FAISS search execution time in milliseconds")
    results: List[SearchResultItem] = Field(..., description="Top K visually similar catalog products")


class IndexRebuildResponse(BaseModel):
    """Response schema for catalog index rebuild status."""

    status: str
    total_indexed: int
    message: str


class ModelInfoResponse(BaseModel):
    """Diagnostic info schema for model and FAISS vector index status."""

    model_type: str
    embedding_dimension: int
    total_indexed_products: int
    metric_type: str
    gpu_status: Dict[str, Any]
