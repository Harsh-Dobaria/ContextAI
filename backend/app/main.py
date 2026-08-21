from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.models.conversation import Conversation
from app.models.message import Message
from app.api.workspaces import router as workspace_router
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.api.chat import router as chat_router
from app.api.auth import router as auth_router
from app.services.bm25_service import bm25_service

from app.database.database import (
    Base,
    engine,
    SessionLocal,
)

from app.models.workspace import Workspace
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

from app.services.vector_service import vector_store


app = FastAPI(
    title="ContextAI",
    version="1.0.0"
)


# Create database tables
Base.metadata.create_all(
    bind=engine
)


# -----------------------------
# Rebuild FAISS when backend starts
# -----------------------------

@app.on_event("startup")
def startup_event():
    db = SessionLocal()

    try:
        try:
            db.execute(
                text(
                    "ALTER TABLE documents "
                    "ADD COLUMN status VARCHAR DEFAULT 'ready'"
                )
            )
            db.commit()
            print(
                "Migration: Added status column to documents table"
            )
        except Exception:
            db.rollback()

        faiss_count = vector_store.rebuild_from_database(db)

        print(
            f"FAISS loaded {faiss_count} vectors"
        )

        bm25_count = bm25_service.build_from_database(db)

        print(
            f"BM25 loaded {bm25_count} chunks"
        )

    finally:
        db.close()

# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# -----------------------------
# Root
# -----------------------------

@app.get("/")
def root():

    return {
        "message": "ContextAI Backend Running 🚀"
    }


# -----------------------------
# Health
# -----------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }



# -----------------------------
# Routers
# -----------------------------

app.include_router(
    workspace_router
)

app.include_router(
    documents_router
)

app.include_router(
    search_router
)

app.include_router(
    chat_router
)

app.include_router(
    auth_router
)