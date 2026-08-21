from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
import json
import os
import shutil

from app.database.database import get_db, SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.workspace import Workspace
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.vector_service import vector_store
from app.api.dependencies import get_current_user
from app.models.user import User
from typing import cast

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"]
)

UPLOAD_DIR = "uploads"
BATCH_SIZE = 20

os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_document_background(
    document_id: int,
    file_path: str
):
    db = SessionLocal()

    try:
        document = (
            db.query(Document)
            .filter(Document.id == document_id)
            .first()
        )

        if not document:
            print(f"Document {document_id} not found")
            return

        print(f"Processing document: {document.name}")

        extracted_text = extract_text_from_pdf(file_path)

        print(
            f"Extracted characters: {len(extracted_text)}"
        )

        document.extracted_text = extracted_text
        db.commit()

        chunks = chunk_text(extracted_text)

        print(
            f"Created {len(chunks)} chunks"
        )

        if not chunks:
            document.status = "failed"
            db.commit()
            print(
                f"No chunks created for document {document_id}"
            )
            return

        embeddings = generate_embeddings(
            chunks,
            batch_size=BATCH_SIZE
        )

        print(
            f"Generated {len(embeddings)} embeddings"
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Expected {len(chunks)} embeddings "
                f"but received {len(embeddings)}"
            )

        for index, (content, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=index,
                content=content,
                embedding=json.dumps(embedding)
            )

            db.add(chunk)

        db.commit()

        print(
            f"Saved {len(chunks)} chunks to database"
        )

        vector_count = vector_store.rebuild_from_database(
            db
        )

        print(
            f"FAISS now contains {vector_count} vectors"
        )

        document.status = "ready"
        db.commit()

        print(
            f"Document {document_id} is READY"
        )

    except Exception as error:
        print(
            f"Error processing document "
            f"{document_id}: {error}"
        )

        db.rollback()

        try:
            document = (
                db.query(Document)
                .filter(Document.id == document_id)
                .first()
            )

            if document:
                document.status = "failed"
                db.commit()

        except Exception as status_error:
            print(
                f"Could not update document status: "
                f"{status_error}"
            )

    finally:
        db.close()


@router.post("/upload/{workspace_id}")
def upload_document(
    workspace_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    workspace = (
        db.query(Workspace)
        .filter(Workspace.id == workspace_id)
        .first()
    )

    if not workspace or workspace.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="Workspace not found"
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File must have a name"
        )

    filename = file.filename

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    new_document = Document(
        name=filename,
        file_path=file_path,
        workspace_id=workspace_id,
        status="processing"
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    document_id = cast(int, new_document.id)

    background_tasks.add_task(
        process_document_background,
        document_id,
        file_path
    )
    
    return {
        "id": new_document.id,
        "name": new_document.name,
        "file_path": new_document.file_path,
        "workspace_id": new_document.workspace_id,
        "status": new_document.status
    }


@router.get("/")
def get_documents(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace or workspace.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    documents = (
        db.query(Document)
        .filter(
            Document.workspace_id == workspace_id
        )
        .order_by(
            Document.id.desc()
        )
        .all()
    )

    return [
        {
            "id": document.id,
            "name": document.name,
            "status": document.status,
            "workspace_id": document.workspace_id
        }
        for document in documents
    ]


@router.get("/{document_id}/status")
def get_document_status(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    document = (
        db.query(Document)
        .filter(
            Document.id == document_id
        )
        .first()
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
        
    workspace = db.query(Workspace).filter(Workspace.id == document.workspace_id).first()
    if not workspace or workspace.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return {
        "id": document.id,
        "name": document.name,
        "status": document.status,
        "workspace_id": document.workspace_id
    }


@router.get("/vector-count")
def vector_count():
    return {
        "vectors": vector_store.index.ntotal
    }


@router.post("/rebuild-vectors")
def rebuild_vectors(
    db: Session = Depends(get_db)
):
    count = vector_store.rebuild_from_database(db)

    return {
        "message": "FAISS index rebuilt",
        "vectors": count
    }