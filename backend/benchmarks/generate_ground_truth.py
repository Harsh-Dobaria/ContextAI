import json
import os
import sys
import time
from pathlib import Path
from typing import cast
from dotenv import load_dotenv
from collections import defaultdict

# Add backend directory to Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.database.database import SessionLocal
from app.models.document_chunk import DocumentChunk
from app.models.document import Document
from app.services.embedding_service import generate_embedding
from app.services.vector_service import vector_store
from app.services.bm25_service import bm25_service

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "questions.json"
OUTPUT_FILE = BASE_DIR / "benchmark_questions_labeled.json"
WORKSPACE_ID = 2
CANDIDATE_K = 20

# Optional LLM Setup
api_key = os.getenv("GEMINI_API_KEY")
client = None
if api_key:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("Gemini API key found. Automated relevance labeling is enabled.")
    except ImportError:
        print("google-genai package not installed. Automated labeling disabled.")
else:
    print("No GEMINI_API_KEY found. Automated labeling disabled.")


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_chunk_relevance(question: str, candidate_chunks: list[dict]) -> list[int]:
    """Uses Gemini to evaluate which chunks actually contain the answer."""
    if not client or not candidate_chunks:
        return []

    context_parts = []
    for c in candidate_chunks:
        context_parts.append(f"--- Chunk ID: {c['chunk_id']} ---\n{c['text']}\n")

    context_str = "\n".join(context_parts)

    prompt = f"""
You are an expert evaluator for a retrieval-augmented generation (RAG) system.
Your task is to identify which document chunks contain the information necessary to answer the given question.

Question:
{question}

Candidate Document Chunks:
{context_str}

Evaluate each chunk carefully. A chunk is "relevant" if it contains facts or context that directly answers the question.
Do not mark chunks relevant just because they contain similar keywords.
If multiple chunks contain the answer, you can select multiple.
If none of the chunks contain the answer, return an empty list.

Return ONLY a JSON array of the relevant chunk IDs (integers). For example: [123, 125]. Do not include any other text or markdown formatting.
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            
            if response.text is None:
                return []
                
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
                
            relevant_ids = json.loads(text)
            if isinstance(relevant_ids, list):
                return [int(cid) for cid in relevant_ids]
            return []
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                if attempt < max_retries - 1:
                    print(f"  Rate limit hit. Retrying in 30s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(30)
                else:
                    print(f"  Rate limit exceeded on final attempt. Skipping automated labeling for this question.")
                    return []
            else:
                print(f"  Error evaluating question: {e}")
                return []
    return []


def main():
    db = SessionLocal()
    try:
        # Initialize BM25 and load DB chunks for validation
        bm25_service.build_from_database(db)
        
        questions = load_questions()
        labeled_results = []
        
        total_questions = len(questions)
        processed_count = 0
        labeled_count = 0
        unlabeled_count = 0

        print(f"\nGround truth generation started. Total questions: {total_questions}")
        
        for index, q_data in enumerate(questions, start=1):
            question_text = q_data["question"]
            q_id = q_data.get("id", index)
            
            print(f"\nProcessing question {index}/{total_questions}")
            print(f"Question: {question_text}")
            
            # 1. Retrieve Broad Candidate Pool
            query_embedding = generate_embedding(question_text)
            
            faiss_results = vector_store.search(query_embedding, top_k=CANDIDATE_K)
            bm25_results = bm25_service.search(question_text, top_k=CANDIDATE_K)
            
            faiss_ids = [cid for cid, _ in faiss_results]
            bm25_ids = [cid for cid, _ in bm25_results]
            
            # 2. Merge and Filter by Workspace
            unique_ids = list(set(faiss_ids + bm25_ids))
            
            # Fetch content and ensure workspace ID is 2
            chunks = (
                db.query(DocumentChunk, Document.workspace_id)
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(DocumentChunk.id.in_(unique_ids))
                .all()
            )
            
            candidate_chunks = []
            candidate_chunk_ids = []
            for chunk, ws_id in chunks:
                if ws_id == WORKSPACE_ID:
                    cid = cast(int, chunk.id)
                    candidate_chunk_ids.append(cid)
                    
                    sources = []
                    if cid in faiss_ids: sources.append("faiss")
                    if cid in bm25_ids: sources.append("bm25")
                    
                    candidate_chunks.append({
                        "chunk_id": cid,
                        "text": cast(str, chunk.content),
                        "sources": sources
                    })
                    
            print(f"FAISS candidates: {len(faiss_ids)}")
            print(f"BM25 candidates: {len(bm25_ids)}")
            print(f"Unique valid candidates: {len(candidate_chunk_ids)}")
            
            # 3. Optional Automated LLM Labeling
            relevant_ids = []
            if client:
                relevant_ids = evaluate_chunk_relevance(question_text, candidate_chunks)
                time.sleep(4) # Respect free tier RPM limit
            
            if relevant_ids:
                labeled_count += 1
            else:
                unlabeled_count += 1
                
            print(f"Generated relevant chunk IDs: {relevant_ids}")
            
            # 4. Save to output structure
            labeled_results.append({
                "id": q_id,
                "question": question_text,
                "candidate_chunk_ids": candidate_chunk_ids,
                "candidate_chunks": candidate_chunks,
                "relevant_chunk_ids": relevant_ids,
                "is_human_verified": False
            })
            
            processed_count += 1
            
            # Save incrementally
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(labeled_results, f, indent=2)
                
            print(f"Saved generated labels for question {index}")

        print("\n" + "=" * 50)
        print("Ground truth generation complete")
        print(f"Total questions: {total_questions}")
        print(f"Questions processed: {processed_count}")
        print(f"Questions with generated relevant chunks: {labeled_count}")
        print(f"Questions requiring manual review: {unlabeled_count}")
        print("=" * 50 + "\n")

    finally:
        db.close()


if __name__ == "__main__":
    main()
