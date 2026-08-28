from sentence_transformers import SentenceTransformer
from langsmith import traceable

# Load the local model on startup
# Using all-mpnet-base-v2 to preserve the 768 dimension size
model = SentenceTransformer("all-mpnet-base-v2")

EMBEDDING_DIMENSION = 768

@traceable(name="generate_embedding", run_type="embedding")
def generate_embedding(text: str) -> list[float]:
    """
    Generate one embedding using the local model.
    Used for search queries.
    """
    # model.encode returns a numpy array, we convert to list of floats
    embedding = model.encode(text)
    return embedding.tolist()

def generate_embeddings(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Generate embeddings for document chunks using the local model.
    """
    if not texts:
        return []
    
    # sentence-transformers handles batching internally if we pass a list
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings.tolist()