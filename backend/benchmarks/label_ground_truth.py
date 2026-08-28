import json
import os
import sys
from pathlib import Path
import time
from typing import cast
from dotenv import load_dotenv
from google import genai

# Add backend directory to Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.database.database import SessionLocal
from app.models.document_chunk import DocumentChunk
from app.services.agents.retrieval_agent import retrieve_documents, preload_chunks
from app.services.bm25_service import bm25_service
from app.services.embedding_service import generate_embedding

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=api_key)

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "questions.json"
WORKSPACE_ID = 2
TOP_CANDIDATES = 10  # Retrieve top 10 to give LLM choices


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_questions(questions):
    with open(QUESTIONS_FILE, "w", encoding="utf-8") as file:
        json.dump(questions, file, indent=2)


def evaluate_chunk_relevance(question: str, chunk_contents: dict[int, str]) -> list[int]:
    """Uses Gemini to evaluate which chunks actually contain the answer."""
    if not chunk_contents:
        return []

    context_parts = []
    for chunk_id, content in chunk_contents.items():
        context_parts.append(f"--- Chunk ID: {chunk_id} ---\n{content}\n")

    context_str = "\n".join(context_parts)

    prompt = f"""
You are an expert evaluator for a retrieval-augmented generation (RAG) system.
Your task is to identify which document chunks contain the information necessary to answer the given question.

Question:
{question}

Candidate Document Chunks:
{context_str}

Evaluate each chunk carefully. A chunk is "relevant" if it contains facts or context that directly answers the question.
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
                print(f"Warning: Model returned no text for question '{question}'. This could be due to safety filters.")
                return []
                
            text = response.text.strip()
            # Ensure it's just the JSON array
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
            if "429" in error_str and attempt < max_retries - 1:
                print(f"Rate limit hit. Retrying in 30 seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(30)
            else:
                print(f"Error evaluating question '{question}': {e}")
                return []
    return []


def main():
    db = SessionLocal()
    try:
        # Initialize BM25 and pre-load chunks
        bm25_service.build_from_database(db)
        preload_chunks()
        
        questions = load_questions()
        
        for i, q_data in enumerate(questions):
            if "is_human_verified" in q_data:
                print(f"Skipping {i+1}/{len(questions)} (already processed).")
                continue
                
            question_text = q_data["question"]
            print(f"\nProcessing {i+1}/{len(questions)}: {question_text}")
            
            # Use hybrid retrieval to get broad candidate set
            query_embedding = generate_embedding(question_text)
            result = retrieve_documents(
                question=question_text,
                workspace_id=WORKSPACE_ID,
                retrieval_count=TOP_CANDIDATES,
                retrieval_mode="hybrid",
                query_embedding=query_embedding
            )
            
            candidate_ids = result.get("chunk_ids", [])
            
            if not candidate_ids:
                print("No candidate chunks retrieved.")
                q_data["relevant_chunk_ids"] = []
                q_data["is_human_verified"] = False
                continue
                
            # Fetch content for candidate IDs
            chunks = db.query(DocumentChunk).filter(DocumentChunk.id.in_(candidate_ids)).all()
            chunk_contents = {cast(int, c.id): cast(str, c.content) for c in chunks}
            
            # Use LLM to filter to actually relevant chunks
            relevant_ids = evaluate_chunk_relevance(question_text, chunk_contents)
            print(f"Candidates: {candidate_ids} -> Relevant: {relevant_ids}")
            
            q_data["relevant_chunk_ids"] = relevant_ids
            # Keep clear distinction between automatically suggested and human-verified
            q_data["is_human_verified"] = False
            
            # Save incrementally in case of crash
            save_questions(questions)
            
            # Sleep briefly to avoid rate limits (15 RPM free tier limit)
            time.sleep(4)
            
        print("\nFinished labeling ground truth. Saved to questions.json.")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
