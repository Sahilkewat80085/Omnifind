from core.document_pipeline.models import ChunkContextObject, NormalizedDocument


class SemanticChunker:
    """Chunks normalized document text blocks by semantic coherence while preserving structural back-references."""

    def __init__(self, target_tokens: int = 384, overlap_tokens: int = 48, min_tokens: int = 32) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    def chunk(self, doc: NormalizedDocument, file_id: str) -> list[ChunkContextObject]:
        chunks: list[ChunkContextObject] = []
        if not doc.blocks:
            return chunks

        current_words: list[str] = []
        current_location: str = doc.blocks[0].location
        current_headings: list[str] = list(doc.blocks[0].heading_path)

        chunk_counter = 0

        def emit_chunk():
            nonlocal chunk_counter, current_words
            if not current_words:
                return

            chunk_counter += 1
            chunk_text = " ".join(current_words).strip()
            heading_ctx = " > ".join(current_headings) if current_headings else None

            chunks.append(
                ChunkContextObject(
                    chunk_id=f"{file_id}-chunk-{chunk_counter:04d}",
                    text=chunk_text,
                    location=current_location,
                    heading_context=heading_ctx,
                    embedding=[],
                    chunk_summary=None,
                )
            )

            # Keep overlap words for context continuity
            if len(current_words) > self.overlap_tokens:
                current_words = current_words[-self.overlap_tokens :]
            else:
                current_words.clear()

        for block in doc.blocks:
            block_words = block.text.split()
            if not block_words:
                continue

            # If a major heading starts, or location changes drastically, emit previous chunk
            if block.block_type == "heading" and len(current_words) >= self.min_tokens:
                emit_chunk()
                current_headings = list(block.heading_path)
                current_location = block.location

            if block.heading_path:
                current_headings = list(block.heading_path)
            current_location = block.location

            for word in block_words:
                current_words.append(word)
                if len(current_words) >= self.target_tokens:
                    emit_chunk()

        if current_words:
            emit_chunk()

        return chunks
