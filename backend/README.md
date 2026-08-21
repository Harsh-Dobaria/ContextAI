# ContextAI Backend

The backend engine for ContextAI — a FastAPI-powered Retrieval-Augmented Generation (RAG) platform featuring hybrid FAISS + BM25 search, Reciprocal Rank Fusion, Cross-Encoder reranking, LangGraph agent orchestration, and automated citation verification.

For the full system documentation, architecture diagrams, and benchmark results, please refer to the [Root README](../README.md).

## Quickstart

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure environment (.env)
# GEMINI_API_KEY=your_key_here

# Run server
uvicorn app.main:app --reload
```

## Running Benchmarks

```bash
python benchmarks/run_benchmark.py
```
