import hashlib
import os
from pathlib import Path
from typing import Any

from core.document_pipeline.models import (
    DocumentContextObject,
    TableContextObject,
)
from core.document_pipeline.normalizer import DocumentNormalizer
from core.document_pipeline.router import FormatRouter
from core.document_pipeline.semantic.chunker import SemanticChunker
from core.document_pipeline.semantic.context_describer import ContextDescriber
from core.document_pipeline.semantic.embeddings import DocumentEmbedder
from core.document_pipeline.semantic.keywords import KeywordExtractor
from core.document_pipeline.semantic.ner import EntityRecognizer
from core.document_pipeline.semantic.summarizer import DocumentSummarizer
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_file_id(file_path: str) -> str:
    """Produces an idempotent SHA-256 hash representing the file content."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DocumentPipeline:
    """End-to-end context extraction pipeline for .txt, .md, .pdf, .docx, .pptx, and .xlsx files."""

    def __init__(self) -> None:
        self.router = FormatRouter()
        self.normalizer = DocumentNormalizer()
        self.chunker = SemanticChunker()
        self.ner = EntityRecognizer()
        self.keywords = KeywordExtractor()
        self.summarizer = DocumentSummarizer()
        self.context_describer = ContextDescriber()
        self.embedder = DocumentEmbedder()

    def extract_context(self, file_path: str) -> dict[str, Any]:
        """Extracts complete structured context object from a single document.

        Args:
            file_path: Path to the target document.

        Returns:
            Dictionary matching the DocumentContextObject schema.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        extractor = self.router.get_extractor(str(path))
        if extractor is None:
            raise ValueError(f"Unsupported document format: {path.suffix}")

        file_id = generate_file_id(str(path))

        # 1. Format-specific extraction
        raw_doc = extractor.extract(str(path))

        # 2. Normalization layer
        norm_doc = self.normalizer.normalize(raw_doc)

        # 3. Concatenate text for document-level semantic analysis
        all_text = "\n\n".join([b.text for b in norm_doc.blocks])
        word_count = len(all_text.split())

        # 4. Semantic Pipeline Stages
        chunks = self.chunker.chunk(norm_doc, file_id)
        entities = self.ner.extract_entities(all_text)
        keywords = self.keywords.extract_keywords(all_text, top_n=10)
        summary = self.summarizer.summarize(all_text)
        context_description = self.context_describer.describe(norm_doc, keywords, summary)

        # 5. Embeddings Generation
        doc_embedding = self.embedder.embed_text(context_description or all_text[:1000])

        # Chunk embeddings
        if chunks:
            chunk_texts = [c.text for c in chunks]
            chunk_embeddings = self.embedder.embed_chunks(chunk_texts)
            for c, emb in zip(chunks, chunk_embeddings):
                c.embedding = emb

        # Table processing & embeddings
        tables_context: list[TableContextObject] = []
        for t in norm_doc.tables:
            table_desc = self.embedder.generate_table_description(
                location=t.location,
                headers=t.headers,
                rows=t.rows,
                caption=t.caption,
            )
            table_emb = self.embedder.embed_text(table_desc)
            structured_data = ([t.headers] if t.headers else []) + t.rows
            tables_context.append(
                TableContextObject(
                    table_id=f"{file_id}-{t.table_id}",
                    location=t.location,
                    structured_data=structured_data,
                    table_description=table_desc,
                    embedding=table_emb,
                )
            )

        # Build final context schema
        context_obj = DocumentContextObject(
            file_id=file_id,
            file_path=norm_doc.file_path,
            file_name=norm_doc.file_name,
            file_type=norm_doc.file_type,
            created_at=norm_doc.created_at.isoformat(),
            modified_at=norm_doc.modified_at.isoformat(),
            language="en",
            word_count=word_count,
            is_scanned=norm_doc.is_scanned,
            format_metadata=norm_doc.format_metadata.model_dump(),
            document_embedding=doc_embedding,
            document_summary=summary,
            context_description=context_description,
            keywords=keywords,
            entities=entities,
            tables=tables_context,
            chunks=chunks,
        )

        return context_obj.model_dump()


def extract_context(file_path: str) -> dict[str, Any]:
    """Convenience functional entry point for extracting document context."""
    pipeline = DocumentPipeline()
    return pipeline.extract_context(file_path)
