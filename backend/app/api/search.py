from typing import cast

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.document_chunk import DocumentChunk
from app.models.document import Document
from app.services.embedding_service import generate_embedding
from app.services.vector_service import vector_store


router = APIRouter(
    prefix="/api/search",
    tags=["Search"]
)


@router.get("/")
def search(
    query: str,
    workspace_id: int,
    db: Session = Depends(get_db)
):

    query_embedding = generate_embedding(
        query
    )

    chunk_ids = vector_store.search(
        query_embedding,
        top_k=50
    )

    if not chunk_ids:

        return {
            "query": query,
            "results": []
        }

    chunks = (
        db.query(DocumentChunk)
        .join(Document, DocumentChunk.document_id == Document.id)
        .filter(
            DocumentChunk.id.in_(chunk_ids),
            Document.workspace_id == workspace_id
        )
        .all()
    )

    chunk_map = {
        cast(int, chunk.id): chunk
        for chunk in chunks
    }

    results = []

    for chunk_id in chunk_ids:

        chunk = chunk_map.get(
            chunk_id
        )

        if chunk:

            results.append({
                "chunk_id": cast(
                    int,
                    chunk.id
                ),
                "document_id": cast(
                    int,
                    chunk.document_id
                ),
                "content": cast(
                    str,
                    chunk.content
                )
            })

            if len(results) == 3:
                break

    return {
        "query": query,
        "results": results
    }