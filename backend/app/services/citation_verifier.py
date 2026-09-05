import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()


api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set"
    )


client = genai.Client(
    api_key=api_key
)


def verify_answer(
    question: str,
    answer: str,
    context: str
) -> bool:

    if not context.strip():
        return False

    prompt = f"""
You are a strict answer verification system.

Your job is to determine whether the answer is fully
supported by the provided context.

Question:
{question}

Answer:
{answer}

Context:
{context}

Rules:
- Use ONLY the provided context.
- Do not use outside knowledge.
- Return YES only if the answer is supported by the context.
- Return NO if the answer contains unsupported,
  invented, or contradictory information.
- Return only YES or NO.

Verification:
"""

    MAX_RETRIES = 5

    response = None

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = (
                client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )
            )

            break

        except (errors.ServerError, errors.ClientError) as error:

            if attempt == MAX_RETRIES - 1:

                print(
                    f"Citation verification failed "
                    f"after {MAX_RETRIES} attempts: "
                    f"{error}"
                )

                return False

            is_rate_limit = "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)
            wait_time = 22 if is_rate_limit else (2 ** attempt)

            print(
                f"Verifier error "
                f"(attempt {attempt + 1}/"
                f"{MAX_RETRIES}): {error}. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(
                wait_time
            )

    if response is None:
        return False

    result = (
        response.text
        or ""
    ).strip().upper()

    print(
        f"Citation verification: "
        f"{result}"
    )

    return result == "YES"