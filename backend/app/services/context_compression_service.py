import re

from app.core.constants import BROAD_KEYWORDS



def compress_context(
    question: str,
    context: str,
    max_sentences: int = 12
) -> str:
    if not context.strip():
        return ""

    question_lower = question.lower().strip()

    is_broad = any(
        keyword in question_lower
        for keyword in BROAD_KEYWORDS
    )

    if is_broad:
        max_sentences = 30

    sentences = re.split(
        r"(?<=[.!?])\s+",
        context.strip()
    )

    if len(sentences) <= max_sentences:
        return context

    question_words = {
        word.lower()
        for word in re.findall(
            r"\b\w+\b",
            question
        )
        if len(word) > 2
    }

    scored_sentences: list[tuple[int, str]] = []

    for sentence in sentences:
        sentence_words = {
            word.lower()
            for word in re.findall(
                r"\b\w+\b",
                sentence
            )
            if len(word) > 2
        }

        score = len(
            question_words & sentence_words
        )

        scored_sentences.append(
            (score, sentence)
        )

    ranked = sorted(
        enumerate(scored_sentences),
        key=lambda item: (
            item[1][0],
            -item[0]
        ),
        reverse=True
    )

    selected_indices = sorted(
        index
        for index, _ in ranked[:max_sentences]
    )

    return " ".join(
        sentences[index]
        for index in selected_indices
    )