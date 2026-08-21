import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)


def generate_hyde_document(question: str) -> str:
    """
    Generates a hypothetical document (answer) to the question.
    This is used for Hypothetical Document Embeddings (HyDE) to improve FAISS recall.
    """
    prompt = f"""
Please write a short, factual paragraph answering the following question.
Write it in the style of an encyclopedia article or reference document.
Even if you are unsure of the exact answer, generate a plausible-sounding
document using relevant vocabulary.

Question:
{question}

Factual Paragraph:
"""
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        hypothetical_doc = response.text

        if hypothetical_doc and hypothetical_doc.strip():
            return hypothetical_doc.strip()

    except Exception as error:
        print(f"HyDE generation failed: {error}")

    # Fallback to the original question if generation fails
    return question
