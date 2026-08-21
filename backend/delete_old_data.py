from app.database.database import SessionLocal

from app.models.document_chunk import DocumentChunk


db = SessionLocal()

try:

    deleted = (
        db.query(
            DocumentChunk
        )
        .delete()
    )

    db.commit()

    print(
        f"Deleted {deleted} old chunks."
    )

finally:

    db.close()