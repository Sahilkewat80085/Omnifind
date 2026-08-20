from core.embeddings.text_embedding_service import TextEmbeddingService


class DocumentEmbedder:
    """Generates dense vector embeddings for documents, chunks, and table descriptions."""

    def __init__(self) -> None:
        self._embedder = TextEmbeddingService()

    def embed_text(self, text: str) -> list[float]:
        if not text:
            return []
        return self._embedder.encode_doc(text)

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
        return self._embedder.encode_passages(chunks)

    def generate_table_description(self, location: str, headers: list[str], rows: list[list[str]], caption: str | None = None) -> str:
        """Produces a rich textual representation of a structured table for semantic embedding."""
        col_str = ", ".join([h for h in headers if h])
        desc = f"Table at {location}. Columns: {col_str}."
        if caption:
            desc = f"{caption}. {desc}"

        row_samples = []
        for r in rows[:6]:
            cells = [f"{h}: {val}" for h, val in zip(headers, r) if h and val]
            if cells:
                row_samples.append(", ".join(cells[:4]))

        if row_samples:
            desc += " Rows: " + " | ".join(row_samples) + "."

        return desc
