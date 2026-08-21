from app.database.database import SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk


db = SessionLocal()

try:

    chunks = (
        db.query(DocumentChunk)
        .order_by(DocumentChunk.id)
        .limit(20)
        .all()
    )

    print("\nFIRST 20 CHUNKS\n")

    for chunk in chunks:

        document = (
            db.query(Document)
            .filter(
                Document.id == chunk.document_id
            )
            .first()
        )

        print(
            f"Chunk ID: {chunk.id}"
        )

        print(
            f"Document ID on chunk: "
            f"{chunk.document_id}"
        )

        print(
            f"Document exists: "
            f"{document is not None}"
        )

        if document:
            print(
                f"Document workspace: "
                f"{document.workspace_id}"
            )

        print("-" * 50)

finally:
    db.close()