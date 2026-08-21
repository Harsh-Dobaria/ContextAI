def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 500
) -> list[str]:

    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks: list[str] = []

    start = 0
    step = chunk_size - overlap

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += step

    return chunks