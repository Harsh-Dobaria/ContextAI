import os
import re

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)


FOLLOW_UP_PATTERNS = [
    r"^what about\b",
    r"^how about\b",
    r"^what is its\b",
    r"^what are its\b",
    r"^how does it\b",
    r"^how do they\b",
    r"^why is it\b",
    r"^why does it\b",
    r"^what does it\b",
    r"^who developed it\b",
    r"^when was it\b",
    r"^tell me more\b",
    r"^explain more\b",
]


def is_follow_up_question(question: str) -> bool:
    question_lower = question.lower().strip()

    return any(
        re.search(pattern, question_lower)
        for pattern in FOLLOW_UP_PATTERNS
    )


def reformulate_query(
    question: str,
    previous_question: str | None = None,
    previous_answer: str | None = None
) -> str:

    if not previous_question:
        return question

    if not is_follow_up_question(question):
        return question

    prompt = f"""
Rewrite the user's latest question into a complete,
standalone search query using the previous conversation.

Rules:
- Return ONLY the rewritten query.
- Preserve the user's original intent.
- Resolve pronouns such as "it", "its", "they", and "them".
- Do not answer the question.
- Do not add information not present in the conversation.
- Keep the rewritten query concise.

Previous user question:
{previous_question}

Previous assistant answer:
{previous_answer or ""}

Latest user question:
{question}

Rewritten query:
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        rewritten_query = response.text

        if rewritten_query and rewritten_query.strip():
            return rewritten_query.strip()

    except Exception as error:
        print(
            f"Query rewriting failed: {error}"
        )

    return question