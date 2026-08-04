from pathlib import Path

from core.embeddings.image_embedding_service import ImageEmbeddingService
from core.embeddings.text_embedding_service import TextEmbeddingService
from core.vectorstore.qdrant_client import SearchHit, VectorService
from models.schemas.search_schemas import DocumentResult, ImageResult, SearchResponse
from utils.config import get_settings


def _normalize_scores(hits: list[SearchHit]) -> list[SearchHit]:
    """bge (text) and CLIP (image) cosine scores live on different scales,
    so raw scores can't be compared across modalities. Min-max normalize
    each partition to [0, 1] before merging so ranking reflects relative
    relevance within each modality, not an artifact of which model produced
    the score."""
    if not hits:
        return hits
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [SearchHit(score=1.0, payload=h.payload) for h in hits]
    return [SearchHit(score=(h.score - lo) / (hi - lo), payload=h.payload) for h in hits]


class SearchService:
    def __init__(self) -> None:
        self._vector_service = VectorService()
        self._text_embedder = TextEmbeddingService()
        self._image_embedder = ImageEmbeddingService()
        self._settings = get_settings()

    def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        k = top_k or self._settings.search_top_k

        text_vector = self._text_embedder.encode_query(query)
        image_vector = self._image_embedder.encode_text(query)

        text_hits = _normalize_scores(self._vector_service.search_text(text_vector, top_k=k))
        image_hits = _normalize_scores(self._vector_service.search_image(image_vector, top_k=k))

        results: list[DocumentResult | ImageResult] = [
            self._to_document_result(hit) for hit in text_hits
        ] + [self._to_image_result(hit) for hit in image_hits]

        results.sort(key=lambda r: r.similarity, reverse=True)
        return SearchResponse(query=query, results=results[:k])

    @staticmethod
    def _to_document_result(hit: SearchHit) -> DocumentResult:
        p = hit.payload
        file_path = Path(p["path"])
        return DocumentResult(
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            similarity=hit.score,
            page_number=p.get("page_number"),
            chunk_text=p["chunk_text"],
            chunk_index=p["chunk_index"],
            exists=file_path.is_file(),
        )

    @staticmethod
    def _to_image_result(hit: SearchHit) -> ImageResult:
        p = hit.payload
        dims = p.get("image_dimensions", {})
        file_path = Path(p["path"])
        return ImageResult(
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            similarity=hit.score,
            width=dims.get("width", 0),
            height=dims.get("height", 0),
            exists=file_path.is_file(),
        )
