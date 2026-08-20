from pathlib import Path

from core.embeddings.image_embedding_service import ImageEmbeddingService
from core.embeddings.text_embedding_service import TextEmbeddingService
from core.query.intent import detect_intent
from core.vectorstore.qdrant_client import SearchHit, VectorService
from database.session import SessionLocal
from models.schemas.file_schemas import FileMetadata, FileType
from models.schemas.search_schemas import CodeResult, DocumentResult, ImageResult, SearchResponse
from services.metadata_service import MetadataService
from utils.config import get_settings


def drop_missing_files(hits: list[SearchHit]) -> list[SearchHit]:
    return [hit for hit in hits if Path(hit.payload.get("path", "")).is_file()]


def best_per_file(results: list[DocumentResult | ImageResult | CodeResult]) -> list[DocumentResult | ImageResult | CodeResult]:
    seen: set[str] = set()
    best: list[DocumentResult | ImageResult | CodeResult] = []
    for result in results:
        if result.file_id not in seen:
            seen.add(result.file_id)
            best.append(result)
    return best


def _calibrate(hits: list[SearchHit], *, floor: float, ceil: float, drop_below_floor: bool) -> list[SearchHit]:
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


class SearchService:
    def __init__(self) -> None:
        self._vector_service = VectorService()
        self._text_embedder = TextEmbeddingService()
        self._image_embedder = ImageEmbeddingService()
        self._settings = get_settings()

    def search(self, query: str, top_k: int | None = None, file_type: FileType | None = None) -> SearchResponse:
        k = top_k or self._settings.search_top_k
        intent = detect_intent(query)
        wants = file_type or intent.file_type
        embed_query = intent.query if wants == intent.file_type else query
        results: list[DocumentResult | ImageResult | CodeResult] = []
        seen_paths: set[str] = set()

        db = SessionLocal()
        try:
            filename_matches = MetadataService(db).search_by_filename(intent.query, file_type=wants)
            for file_meta, score in filename_matches:
                if not Path(file_meta.path).is_file() or file_meta.path in seen_paths:
                    continue
                seen_paths.add(file_meta.path)
                results.append(self._filename_result(file_meta, score))
        finally:
            db.close()

        if wants in (None, FileType.document, FileType.code):
            query_vector = self._text_embedder.encode_query(embed_query)
            candidate_limit = k * self._settings.search_candidate_factor
            bands = (
                (FileType.document, self._settings.search_text_score_floor, self._settings.search_text_score_ceil, False),
                (FileType.code, self._settings.search_code_score_floor, self._settings.search_code_score_ceil, True),
            )
            for target_type, floor, ceil, drop in bands:
                if wants not in (None, target_type):
                    continue
                hits = self._vector_service.search_text(query_vector, top_k=candidate_limit, file_type=target_type.value)
                for hit in _calibrate(drop_missing_files(hits), floor=floor, ceil=ceil, drop_below_floor=drop):
                    result = self._to_document_result(hit) if target_type == FileType.document else self._to_code_result(hit)
                    existing = next((item for item in results if item.path == result.path), None)
                    if existing is None:
                        results.append(result)
                    elif isinstance(existing, DocumentResult) and isinstance(result, DocumentResult) and existing.chunk_text.startswith("File:"):
                        existing.chunk_text, existing.page_number, existing.chunk_index = result.chunk_text, result.page_number, result.chunk_index
                    elif isinstance(existing, CodeResult) and isinstance(result, CodeResult) and existing.chunk_text.startswith("File:"):
                        existing.chunk_text, existing.symbol, existing.line_start, existing.line_end = result.chunk_text, result.symbol, result.line_start, result.line_end

        if wants in (None, FileType.image):
            hits = self._vector_service.search_image(self._image_embedder.encode_text(embed_query), top_k=k)
            for hit in _calibrate(drop_missing_files(hits), floor=self._settings.search_image_score_floor, ceil=self._settings.search_image_score_ceil, drop_below_floor=True):
                result = self._to_image_result(hit)
                if not any(item.path == result.path for item in results):
                    results.append(result)

        results.sort(key=lambda result: result.similarity, reverse=True)
        return SearchResponse(query=query, results=best_per_file(results)[:k], filtered_to=wants)

    @staticmethod
    def _filename_result(file_meta: FileMetadata, score: float) -> DocumentResult | ImageResult | CodeResult:
        if file_meta.file_type == FileType.document:
            return DocumentResult(file_id=file_meta.id, file_name=file_meta.file_name, path=file_meta.path, similarity=score, page_number=1, chunk_text=f"File: {file_meta.file_name}", chunk_index=0)
        if file_meta.file_type == FileType.image:
            return ImageResult(file_id=file_meta.id, file_name=file_meta.file_name, path=file_meta.path, similarity=score, width=file_meta.image_width or 0, height=file_meta.image_height or 0)
        return CodeResult(file_id=file_meta.id, file_name=file_meta.file_name, path=file_meta.path, similarity=score, language=file_meta.language or "code", symbol=None, line_start=1, line_end=1, chunk_text=f"File: {file_meta.file_name}", chunk_index=0)

    @staticmethod
    def _to_document_result(hit: SearchHit) -> DocumentResult:
        p = hit.payload
        return DocumentResult(file_id=p["file_id"], file_name=p["file_name"], path=p["path"], similarity=hit.score, page_number=p.get("page_number"), chunk_text=p["chunk_text"], chunk_index=p["chunk_index"])

    @staticmethod
    def _to_code_result(hit: SearchHit) -> CodeResult:
        p = hit.payload
        return CodeResult(file_id=p["file_id"], file_name=p["file_name"], path=p["path"], similarity=hit.score, language=p.get("language", "text"), symbol=p.get("symbol"), line_start=p.get("line_start", 1), line_end=p.get("line_end", 1), chunk_text=p["chunk_text"], chunk_index=p["chunk_index"])

    @staticmethod
    def _to_image_result(hit: SearchHit) -> ImageResult:
        p = hit.payload
        dims = p.get("image_dimensions") or {}
        return ImageResult(file_id=p["file_id"], file_name=p["file_name"], path=p["path"], similarity=hit.score, width=dims.get("width", 0), height=dims.get("height", 0))