import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from app.services.context_compression_service import compress_context


load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")


client = genai.Client(
    api_key=api_key
)


def generate_response(
    question: str,
    context: str
) -> dict[str, Any]:

    if not context.strip():
        return {
            "answer": (
                "I couldn't find that information "
                "in the uploaded documents."
            ),
            "sources_used": 0
        }

    # -----------------------------
    # Context Compression
    # -----------------------------

    compressed_context = compress_context(
        question=question,
        context=context
    )

    print(
        f"Original context characters: "
        f"{len(context)}"
    )

    print(
        f"Compressed context characters: "
        f"{len(compressed_context)}"
    )

    # -----------------------------
    # Prompt
    # -----------------------------

    prompt = f"""
You are ContextAI, a document question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say exactly:

"I couldn't find that information in the uploaded documents."

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- Do not assume information that is not present.
- For questions asking for ALL, EVERY, EACH, LIST, COMPLETE,
  or similar broad requests, include all relevant information
  available in the provided context.
- Do not return only the first matching item when multiple
  relevant items are present.
- Keep the answer clear and concise.

Context:
{compressed_context}

Question:
{question}

Answer:
"""

    # -----------------------------
    # Gemini Request with Retry
    # -----------------------------

    MAX_RETRIES = 3

    response = None

    for attempt in range(MAX_RETRIES):

        try:

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            break

        except errors.ServerError as error:

            if attempt == MAX_RETRIES - 1:

                print(
                    f"Gemini failed after "
                    f"{MAX_RETRIES} attempts: "
                    f"{error}"
                )

                return {
                    "answer": (
                        "The AI service is temporarily "
                        "unavailable. Please try again later."
                    ),
                    "sources_used": len(
                        context.split("\n\n")
                    )
                }

            wait_time = 2 ** attempt

            print(
                f"Gemini unavailable "
                f"(attempt {attempt + 1}/"
                f"{MAX_RETRIES}). "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    # -----------------------------
    # Process Response
    # -----------------------------

    if response is None:
        return {
            "answer": (
                "The AI service is temporarily "
                "unavailable. Please try again later."
            ),
            "sources_used": len(
                context.split("\n\n")
            )
        }

    answer = response.text

    if answer is None or not answer.strip():
        answer = "I couldn't generate an answer."

    return {
        "answer": answer,
        "sources_used": len(
            context.split("\n\n")
        )
    }