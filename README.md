<h1>⚡ DeepVision Search</h1>
<h3>A Production-Grade Deep Learning Visual Product Search Engine</h3>

<p>DeepVision Search is an end-to-end, industry-level Visual Product Similarity Engine built exclusively with <strong>TensorFlow 2.x / Keras</strong>, <strong>FAISS vector indexing</strong>, <strong>FastAPI</strong>, and <strong>Streamlit</strong>. </p>

<p>Instead of traditional classification, this system relies on high-dimensional metric learning representations (128-D L2-normalized embeddings) to retrieve visually similar products from catalogs in real-time.</p>

<hr>

<h2>📐 Architecture Overview</h2>

<pre><code class="language-mermaid">
graph TD
    User([User / Web App / REST API]) -->|Upload Image| Input[Image Preprocessing Pipeline]
    Input -->|Resized & Normalized Image| Model[Deep Neural Net Feature Extractor]
    Model -->|128-D Unit L2 Normalized Vector| Generator[Embedding Generator]
    Generator -->|Query Vector| FAISS[FAISS Vector Store]
    FAISS &lt;--&gt;|Metadata Match| SQLite[(SQLite Metadata DB)]
    FAISS -->|Top-5 Visual Similarity Matches| User
</code></pre>

<hr>

<h2>📁 Directory Structure</h2>

<pre><code>
DeepVisionSearch/
├── app/
│   ├── main.py              # FastAPI REST server endpoints
│   └── schemas.py           # Pydantic request/response schemas
├── frontend/
│   └── app.py               # Streamlit interactive web dashboard
├── models/
│   ├── cnn_scratch.py       # Model Version 1: Custom CNN from scratch
│   ├── transfer_learning.py # Model Version 2: EfficientNetB0 Transfer Learning
│   └── factory.py           # Abstract model factory initializer
├── training/
│   ├── trainer.py           # Training engine with callbacks & mixed precision
│   └── losses.py            # Categorical CrossEntropy, Contrastive & Triplet Loss
├── evaluation/
│   ├── evaluator.py         # Precision, Recall, F1, Top-K accuracy, Confusion Matrix
│   ├── gradcam.py           # Grad-CAM visual explainability heatmap generator
│   └── tsne.py              # t-SNE 2D/3D embedding space visualizer
├── preprocessing/
│   ├── augmentations.py     # tf.keras GPU augmentation pipeline
│   └── dataset.py           # tf.data pipeline ETL loader
├── embeddings/
│   ├── generator.py         # Batch feature extraction engine
│   └── vector_store.py      # FAISS vector store & SQLite manager
├── utils/
│   ├── config.py            # Centralized hyperparameter configuration
│   ├── logger.py            # Structured logging setup
│   ├── gpu.py               # GPU memory growth & mixed precision helper
│   ├── seed.py              # Reproducible seed initialization
│   └── checkpoint.py        # Model weight saving and loading
├── data/
│   ├── synthetic_generator.py # Synthetic product catalog dataset generator
│   ├── raw/                 # Catalog image folders
│   └── index/               # FAISS binary vector index storage
├── weights/                 # Model checkpoint files (.h5)
├── tests/
│   └── test_pipeline.py     # Pytest unit & integration test suite
├── Dockerfile               # Container packaging specification
├── requirements.txt         # Versioned Python dependencies
└── README.md                # Project documentation
</code></pre>

<hr>

<h2>🛠️ Quickstart Installation</h2>

<h3>1. Clone &amp; Set Up Environment</h3>
<pre><code class="language-bash">
git clone https://github.com/your-username/DeepVisionSearch.git
cd DeepVisionSearch

# Create Python virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
</code></pre>

<h3>2. Generate Synthetic Product Catalog (Optional)</h3>
<p>If you do not have a local dataset ready, generate synthetic product categories (shoes, shirts, watches, bags, sunglasses):</p>
<pre><code class="language-bash">
python data/synthetic_generator.py
</code></pre>

<hr>

<h2>🚀 Running the Web Applications</h2>

<h3>Launch Streamlit Dashboard</h3>
<pre><code class="language-bash">
streamlit run frontend/app.py
</code></pre>
<p>Open your browser at <code>http://localhost:8501</code>.</p>

<h3>Launch FastAPI Server</h3>
<pre><code class="language-bash">
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
</code></pre>
<p>Interactive API docs available at <code>http://localhost:8000/docs</code>.</p>

<hr>

<h2>📡 REST API Specifications</h2>

<h3><code>POST /search</code></h3>
<p>Accepts an image upload and returns Top-K visual matches.</p>

<h4>Request Example (cURL)</h4>
<pre><code class="language-bash">
curl -X POST "http://localhost:8000/search?top_k=5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@query_shoe.jpg"
</code></pre>

<h4>JSON Response Format</h4>
<pre><code class="language-json">
{
  "inference_time_ms": 14.85,
  "results": [
    {
      "filename": "shoes_001.jpg",
      "similarity": 98.42,
      "category": "shoes",
      "filepath": "C:\\DeepVisionSearch\\data\\raw\\shoes\\shoes_001.jpg",
      "raw_score": 0.9842
    },
    {
      "filename": "shoes_004.jpg",
      "similarity": 92.15,
      "category": "shoes",
      "filepath": "C:\\DeepVisionSearch\\data\\raw\\shoes\\shoes_004.jpg",
      "raw_score": 0.9215
    }
  ]
}
</code></pre>

<hr>

<h2>🧪 Running Automated Unit Tests</h2>
<p>Run pytest to verify model shapes, FAISS indexing, dataset loading, and loss functions:</p>
<pre><code class="language-bash">
pytest tests/test_pipeline.py -v
</code></pre>

<hr>

<h2>🐳 Docker Deployment</h2>
<p>Build and run the containerized application using Docker:</p>
<pre><code class="language-bash">
# Build Docker image
docker build -t deepvision-search:latest .

# Run container
docker run -p 8000:8000 -p 8501:8501 deepvision-search:latest
</code></pre>

<hr>

<h2>🔮 Future Improvements</h2>
<ol>
  <li><strong>Product Segmentation:</strong> Integrate Segment Anything (SAM) or YOLO object detection to isolate products from complex background scenes prior to feature embedding extraction.</li>
  <li><strong>HNSW FAISS Index:</strong> Scale up index searching for multi-million catalog sizes using <code>IndexHNSWFlat</code>.</li>
  <li><strong>Multi-Modal Retrieval:</strong> Incorporate vision-language representations (e.g. CLIP/ALIGN via TensorFlow) to support hybrid text + image search.</li>
</ol>
