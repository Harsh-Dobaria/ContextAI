from app.database.database import SessionLocal
from app.models.document_chunk import DocumentChunk

db = SessionLocal()

chunks = db.query(DocumentChunk).all()

print(f"\nTotal chunks in database: {len(chunks)}\n")

for chunk in chunks[:3]:
    print("Chunk ID:", chunk.id)
    print("Content preview:")
    print(chunk.content[:300])
    print("-" * 50)

db.close()