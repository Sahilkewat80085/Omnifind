from pathlib import Path
from typing import Sequence

from core.embeddings.image_embedding_service import ImageEmbeddingService
from core.embeddings.text_embedding_service import TextEmbeddingService
from core.query.intent import detect_intent
from core.vectorstore.qdrant_client import SearchHit, VectorService
from database.session import SessionLocal
from models.schemas.file_schemas import FileMetadata, FileType
from models.schemas.search_schemas import (
    CodeResult,
    DocumentResult,
    ImageResult,
    SearchResponse,
    SearchResult,
)
from services.metadata_service import MetadataService
from utils.config import get_settings


def drop_missing_files(hits: list[SearchHit]) -> list[SearchHit]:
    """Discard hits whose file is no longer on disk."""
    return [hit for hit in hits if Path(hit.payload.get("path", "")).is_file()]


def _calibrate(
    hits: list[SearchHit], *, floor: float, ceil: float, drop_below_floor: bool
) -> list[SearchHit]:
    """Map one modality's raw cosine scores onto a shared 0-1 scale."""
    span = ceil - floor
    if span <= 0:
        return hits

    calibrated = []
    for hit in hits:
        score = (hit.score - floor) / span
        if score <= 0.0:
            if drop_below_floor:
                continue
            score = 0.0
        calibrated.append(SearchHit(score=min(score, 1.0), payload=hit.payload))
    return calibrated


def best_per_file(results: Sequence[SearchResult]) -> list[SearchResult]:
    """Collapse ranked chunks to one row per file, preserving original order."""
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for r in results:
        if r.file_id in seen:
            continue
        seen.add(r.file_id)
        deduped.append(r)
    return deduped


class SearchService:
    def __init__(self) -> None:
        self._vector_service = VectorService()
        self._text_embedder = TextEmbeddingService()
        self._image_embedder = ImageEmbeddingService()
        self._settings = get_settings()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        file_type: FileType | None = None,
    ) -> SearchResponse:
        # Default to a generous pool so users searching across large repositories
        # see all matching files rather than having them cut off after 10.
        k = top_k or 200

        intent = detect_intent(query)
        wants = file_type if file_type is not None else intent.file_type
        search_query = intent.query if file_type is None else query

        results: list[DocumentResult | ImageResult | CodeResult] = []
        seen_paths: set[str] = set()

        # 1. Exact & Fuzzy Filename & Language Matches (100% precision / keyword search)
        db = SessionLocal()
        try:
            meta_service = MetadataService(db)
            filename_matches = meta_service.search_by_filename(search_query, file_type=wants)
            for file_meta, score in filename_matches:
                if not Path(file_meta.path).is_file():
                    continue
                if file_meta.path in seen_paths:
                    continue
                seen_paths.add(file_meta.path)

                if file_meta.file_type == FileType.document:
                    results.append(
                        DocumentResult(
                            file_id=file_meta.id,
                            file_name=file_meta.file_name,
                            path=file_meta.path,
                            similarity=score,
                            page_number=1,
                            chunk_text=f"File: {file_meta.file_name}",
                            chunk_index=0,
                        )
                    )
                elif file_meta.file_type == FileType.image:
                    results.append(
                        ImageResult(
                            file_id=file_meta.id,
                            file_name=file_meta.file_name,
                            path=file_meta.path,
                            similarity=score,
                            width=file_meta.image_width or 0,
                            height=file_meta.image_height or 0,
                        )
                    )
                elif file_meta.file_type == FileType.code:
                    results.append(
                        CodeResult(
                            file_id=file_meta.id,
                            file_name=file_meta.file_name,
                            path=file_meta.path,
                            similarity=score,
                            language=file_meta.language or "code",
                            symbol=None,
                            line_start=1,
                            line_end=1,
                            chunk_text=f"File: {file_meta.file_name}",
                            chunk_index=0,
                        )
                    )
        finally:
            db.close()

        # 2. Semantic Vector Search across Document & Code Partitions
        if wants in (None, FileType.document, FileType.code):
            query_vector = self._text_embedder.encode_query(search_query)

            if wants in (None, FileType.document):
                doc_hits = drop_missing_files(
                    self._vector_service.search_text(
                        query_vector, top_k=k * 4, file_type=FileType.document.value
                    )
                )
                for hit in _calibrate(
                    doc_hits,
                    floor=self._settings.search_text_score_floor,
                    ceil=self._settings.search_text_score_ceil,
                    drop_below_floor=False,
                ):
                    doc_res = self._to_document_result(hit)
                    existing = next((r for r in results if r.path == doc_res.path), None)
                    if existing is None:
                        results.append(doc_res)
                    elif isinstance(existing, DocumentResult) and existing.chunk_text.startswith("File:"):
                        existing.chunk_text = doc_res.chunk_text
                        existing.page_number = doc_res.page_number
                        existing.chunk_index = doc_res.chunk_index

            if wants in (None, FileType.code):
                code_hits = drop_missing_files(
                    self._vector_service.search_text(
                        query_vector, top_k=k * 4, file_type=FileType.code.value
                    )
                )
                for hit in _calibrate(
                    code_hits,
                    floor=self._settings.search_code_score_floor,
                    ceil=self._settings.search_code_score_ceil,
                    drop_below_floor=True,
                ):
                    code_res = self._to_code_result(hit)
                    existing = next((r for r in results if r.path == code_res.path), None)
                    if existing is None:
                        results.append(code_res)
                    elif isinstance(existing, CodeResult) and existing.chunk_text.startswith("File:"):
                        existing.chunk_text = code_res.chunk_text
                        existing.symbol = code_res.symbol
                        existing.line_start = code_res.line_start
                        existing.line_end = code_res.line_end

        # 3. Semantic Vector Search across Image Partition
        if wants in (None, FileType.image):
            image_hits = drop_missing_files(
                self._vector_service.search_image(
                    self._image_embedder.encode_text(search_query), top_k=k * 4
                )
            )
            for hit in _calibrate(
                image_hits,
                floor=self._settings.search_image_score_floor,
                ceil=self._settings.search_image_score_ceil,
                drop_below_floor=True,
            ):
                img_res = self._to_image_result(hit)
                if not any(r.path == img_res.path for r in results):
                    results.append(img_res)

        # 4. Sort all results by similarity descending and collapse per file
        results.sort(key=lambda r: r.similarity, reverse=True)
        deduped = best_per_file(results)
        return SearchResponse(query=query, results=deduped if top_k is None else deduped[:k], filtered_to=wants)

    @staticmethod
    def _to_document_result(hit: SearchHit) -> DocumentResult:
        p = hit.payload
        return DocumentResult(
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            similarity=hit.score,
            page_number=p.get("page_number"),
            chunk_text=p["chunk_text"],
            chunk_index=p["chunk_index"],
        )

    @staticmethod
    def _to_code_result(hit: SearchHit) -> CodeResult:
        p = hit.payload
        return CodeResult(
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            similarity=hit.score,
            language=p.get("language", "text"),
            symbol=p.get("symbol"),
            line_start=p.get("line_start", 1),
            line_end=p.get("line_end", 1),
            chunk_text=p["chunk_text"],
            chunk_index=p["chunk_index"],
        )

    @staticmethod
    def _to_image_result(hit: SearchHit) -> ImageResult:
        p = hit.payload
        dims = p.get("image_dimensions") or {}
        return ImageResult(
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            similarity=hit.score,
            width=dims.get("width", 0),
            height=dims.get("height", 0),
        )
