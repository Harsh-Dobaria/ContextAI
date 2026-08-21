from typing import Any

from app.core.constants import BROAD_KEYWORDS


KEYWORD_INDICATORS = [
    "exactly",
    "definition",
    "define",
    "syntax",
    "stands for",
    "meaning",
    "difference",
    "compare",
]

QUESTION_WORDS = [
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
]


def analyze_query(question: str) -> dict[str, Any]:
    question_lower = question.lower().strip()

    is_broad = any(
        keyword in question_lower
        for keyword in BROAD_KEYWORDS
    )

    is_keyword_focused = any(
        indicator in question_lower
        for indicator in KEYWORD_INDICATORS
    )

    has_question_word = any(
        word in question_lower.split()
        for word in QUESTION_WORDS
    )

    if is_broad:
        retrieval_count = 10
        retrieval_strategy = "broad"

    elif is_keyword_focused:
        retrieval_count = 3
        retrieval_strategy = "keyword"

    elif has_question_word:
        retrieval_count = 3
        retrieval_strategy = "semantic"

    else:
        retrieval_count = 3
        retrieval_strategy = "semantic"

    return {
        "question": question,
        "is_broad": is_broad,
        "is_keyword_focused": is_keyword_focused,
        "intent": "document_lookup",
        "retrieval_count": retrieval_count,
        "retrieval_strategy": retrieval_strategy,
    }