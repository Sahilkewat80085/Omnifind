from pathlib import Path

from core.embeddings.image_embedding_service import ImageEmbeddingService
from core.embeddings.text_embedding_service import TextEmbeddingService
from core.query.intent import detect_intent
from core.vectorstore.qdrant_client import SearchHit, VectorService
from models.schemas.file_schemas import FileType
from models.schemas.search_schemas import CodeResult, DocumentResult, ImageResult, SearchResponse
from utils.config import get_settings


def _calibrate(
    hits: list[SearchHit], *, floor: float, ceil: float, drop_below_floor: bool
) -> list[SearchHit]:
    """Map one modality's raw cosine scores onto a shared 0-1 scale.

    bge (text) and CLIP (image) cosine scores live on different ranges, so raw
    scores can't be compared across modalities. Each modality is rescaled
    through its own measured noise floor and strong-match ceiling, which keeps
    the absolute strength of a match — unlike min-max over the returned hits,
    which always pins the best hit to 1.0 however weak it actually is.

    `drop_below_floor` discards sub-noise hits instead of clamping them; used
    for images, where an unrelated picture is an obvious wrong answer.
    """
    span = ceil - floor
    if span <= 0:  # misconfigured; fall back to raw scores rather than divide by zero
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

    def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        k = top_k or self._settings.search_top_k

        # "mountain image" names the type it wants. Handing back a PDF that
        # happens to discuss mountains is a wrong answer however strong the
        # semantic match, so a stated type becomes a filter on which
        # partitions are searched at all — not a ranking hint applied after.
        intent = detect_intent(query)
        wants = intent.file_type
        results: list[DocumentResult | ImageResult | CodeResult] = []

        if wants in (None, FileType.document, FileType.code):
            # Documents and code share the text partition, so one type can
            # crowd the other out of a top-k before either is scored.
            # Over-fetching keeps both represented; results[:k] still trims.
            raw_text_hits = self._vector_service.search_text(
                self._text_embedder.encode_query(intent.query), top_k=k * 3
            )
            is_code = lambda hit: hit.payload.get("file_type") == FileType.code.value  # noqa: E731

            if wants in (None, FileType.document):
                results += [
                    self._to_document_result(hit)
                    for hit in _calibrate(
                        [h for h in raw_text_hits if not is_code(h)],
                        floor=self._settings.search_text_score_floor,
                        ceil=self._settings.search_text_score_ceil,
                        drop_below_floor=False,
                    )
                ]

            if wants in (None, FileType.code):
                # Its own band, and dropped rather than clamped — see the
                # measurements behind search_code_score_* in Settings.
                results += [
                    self._to_code_result(hit)
                    for hit in _calibrate(
                        [h for h in raw_text_hits if is_code(h)],
                        floor=self._settings.search_code_score_floor,
                        ceil=self._settings.search_code_score_ceil,
                        drop_below_floor=True,
                    )
                ]

        if wants in (None, FileType.image):
            # Reached only when images are actually wanted, so a code or
            # document query never pays to load OpenCLIP.
            results += [
                self._to_image_result(hit)
                for hit in _calibrate(
                    self._vector_service.search_image(
                        self._image_embedder.encode_text(intent.query), top_k=k
                    ),
                    floor=self._settings.search_image_score_floor,
                    ceil=self._settings.search_image_score_ceil,
                    drop_below_floor=True,
                )
            ]

        results.sort(key=lambda r: r.similarity, reverse=True)
        # `query` stays as typed — the UI echoes it back, and showing the
        # stripped version would look like the search misread the question.
        return SearchResponse(query=query, results=results[:k], filtered_to=wants)

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
            exists=Path(p["path"]).is_file(),
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
            exists=Path(p["path"]).is_file(),
        )

    @staticmethod
    def _to_image_result(hit: SearchHit) -> ImageResult:
        p = hit.payload
        dims = p.get("image_dimensions", {})
        return ImageResult(
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            similarity=hit.score,
            width=dims.get("width", 0),
            height=dims.get("height", 0),
            exists=Path(p["path"]).is_file(),
        )
