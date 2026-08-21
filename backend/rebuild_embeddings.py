import json
import time
import os
import sys
from pathlib import Path

# Add backend directory to Python path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from app.database.database import SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.embedding_service import generate_embeddings
from app.services.vector_service import vector_store

def main():
    print("Starting re-embedding process...")
    db = SessionLocal()
    try:
        chunks = db.query(DocumentChunk).all()
        print(f"Found {len(chunks)} chunks to re-embed.")
        
        if not chunks:
            print("No chunks found.")
            return

        texts = [str(chunk.content) for chunk in chunks]
        
        print("Generating embeddings using local model...")
        t0 = time.perf_counter()
        embeddings = generate_embeddings(texts, batch_size=32)
        print(f"Generated {len(embeddings)} embeddings in {time.perf_counter() - t0:.2f}s.")
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = json.dumps(emb)  # type: ignore
            
        print("Committing to database...")
        db.commit()
        
        print("Rebuilding FAISS index...")
        vector_store.rebuild_from_database(db)
        print("Done!")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
