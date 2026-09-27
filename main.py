

import time
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from config.config import config
from models.factory import ModelFactory
from preprocessing.dataset import ProductDatasetPipeline
from embeddings.generator import EmbeddingGenerator
from embeddings.vector_store import FAISSVectorStore
from data.synthetic_generator import create_synthetic_catalog
from utils.gpu import get_device_info, setup_gpu_environment
from utils.logger import get_logger
from app.schemas import SearchResponse, SearchResultItem, ModelInfoResponse, IndexRebuildResponse

logger = get_logger(__name__)

# Create FastAPI application instance
app = FastAPI(
    title="DeepVision Search API",
    description="High-performance Deep Learning Visual Product Similarity Search Engine API powered by TensorFlow & FAISS.",
    version="1.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global application runtime state
state: Dict[str, Any] = {}


@app.on_event("startup")
async def startup_event():
    """Initializes models, pipeline engines, and vector index upon API server boot."""
    logger.info("Initializing DeepVision Search API runtime environment...")
    setup_gpu_environment(config.training.mixed_precision)

    # Instantiate model via factory
    model, _ = ModelFactory.create_model(config.model)
    dataset_pipeline = ProductDatasetPipeline(image_size=config.model.image_size)
    embedding_generator = EmbeddingGenerator(model=model, batch_size=config.training.batch_size)

    # Initialize FAISS Vector Store
    vector_store = FAISSVectorStore(
        embedding_dim=config.model.embedding_dim,
        metric_type=config.index.metric_type,
        db_path=config.paths.db_path,
        index_file_path=config.paths.index_file_path
    )

    # Load existing FAISS index or create initial index if catalog exists
    if not vector_store.load_index():
        logger.info("No pre-existing FAISS index found. Checking for catalog data...")
        if not any(config.paths.raw_data_dir.iterdir()):
            logger.info("Generating synthetic catalog images for initialization...")
            create_synthetic_catalog(config.paths.raw_data_dir)

        dataset, image_paths, labels = dataset_pipeline.build_dataset_from_directory(
            directory=config.paths.raw_data_dir,
            is_training=False
        )
        if image_paths:
            embeddings, metadata = embedding_generator.extract_batch_embeddings(dataset, image_paths)
            vector_store.add_embeddings(embeddings, metadata)
            vector_store.save_index()

    state["model"] = model
    state["pipeline"] = dataset_pipeline
    state["generator"] = embedding_generator
    state["vector_store"] = vector_store
    logger.info("DeepVision Search API initialized successfully and ready for queries.")


@app.get("/", tags=["Health"])
async def root():
    """Root health check endpoint."""
    return {"status": "online", "system": "DeepVision Search API", "version": "1.0.0"}


@app.get("/info", response_model=ModelInfoResponse, tags=["Diagnostic"])
async def get_model_info():
    """Returns runtime model architecture parameters and FAISS vector index status."""
    vector_store: FAISSVectorStore = state["vector_store"]
    return ModelInfoResponse(
        model_type=config.model.model_type,
        embedding_dimension=config.model.embedding_dim,
        total_indexed_products=vector_store.index.ntotal,
        metric_type=config.index.metric_type,
        gpu_status=get_device_info()
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_similar_products(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of top visual matches to return")
):
    """Accepts an uploaded product query image and returns top-K visually similar catalog items.

    Args:
        file: Uploaded image file (JPEG, PNG).
        top_k: Number of similarity matches requested.

    Returns:
        SearchResponse: JSON containing top match items with similarity percentages and latency.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    start_time = time.time()
    try:
        contents = await file.read()
        dataset_pipeline: ProductDatasetPipeline = state["pipeline"]
        generator: EmbeddingGenerator = state["generator"]
        vector_store: FAISSVectorStore = state["vector_store"]

        # Process image and extract query vector
        img_tensor = dataset_pipeline.load_single_image(contents)
        query_embedding = generator.extract_single_embedding(img_tensor)

        # Search FAISS index
        matches = vector_store.search(query_embedding, top_k=top_k)

        elapsed_ms = (time.time() - start_time) * 1000.0

        results = [
            SearchResultItem(
                filename=m["filename"],
                similarity=m["similarity"],
                category=m["category"],
                filepath=m["filepath"],
                raw_score=m["raw_score"]
            )
            for m in matches
        ]

        return SearchResponse(
            inference_time_ms=round(elapsed_ms, 2),
            results=results
        )

    except Exception as e:
        logger.error(f"Error executing similarity search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Search Engine Error: {str(e)}")


@app.post("/index/rebuild", response_model=IndexRebuildResponse, tags=["Index Management"])
async def rebuild_index():
    """Triggers re-scanning of raw image directory, extracts embeddings, and updates FAISS index."""
    try:
        dataset_pipeline: ProductDatasetPipeline = state["pipeline"]
        generator: EmbeddingGenerator = state["generator"]
        vector_store: FAISSVectorStore = state["vector_store"]

        dataset, image_paths, labels = dataset_pipeline.build_dataset_from_directory(
            directory=config.paths.raw_data_dir,
            is_training=False
        )

        if not image_paths:
            raise HTTPException(status_code=404, detail="No catalog images found to index.")

        # Reset existing index
        vector_store.index.reset()
        embeddings, metadata = generator.extract_batch_embeddings(dataset, image_paths)
        vector_store.add_embeddings(embeddings, metadata)
        vector_store.save_index()

        return IndexRebuildResponse(
            status="success",
            total_indexed=vector_store.index.ntotal,
            message="FAISS vector store successfully rebuilt and synchronized."
        )
    except Exception as e:
        logger.error(f"Failed to rebuild vector index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
