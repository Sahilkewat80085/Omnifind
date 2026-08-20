import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from sqlalchemy.orm import Session

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
from services.metadata_service import MetadataService, normalize_content_for_dedup
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


def is_exact_term_query(query: str) -> bool:
    """Classifies whether a query represents an exact term, acronym, code, or quoted literal."""
    q_str = query.strip()
    if not q_str:
        return False

    # 1. Quoted exact search e.g. "DealBridge"
    if q_str.startswith('"') and q_str.endswith('"') and len(q_str) > 2:
        return True

    words = q_str.split()
    if len(words) > 2:
        return False

    # 2. All-caps acronym / acronym code e.g. AWS, DBMS, SQL, API, ML
    if q_str.isupper() and len(q_str) >= 2:
        return True

    # 3. Short single term (<= 5 chars) without whitespace e.g. "aws", "jwt", "cve"
    if len(words) == 1 and len(q_str) <= 5 and re.match(r"^[a-zA-Z0-9_-]+$", q_str):
        return True

    # 4. Alphanumeric identifier e.g. "v1.5", "G2_Review", "404"
    if any(c.isdigit() for c in q_str) and any(c.isalpha() for c in q_str):
        return True

    return False


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


def collapse_near_duplicates(results: list[SearchResult]) -> list[SearchResult]:
    """Collapses identical or near-identical text passages across different files into a single representative card."""
    seen_hashes: dict[str, SearchResult] = {}
    collapsed: list[SearchResult] = []

    for r in results:
        if isinstance(r, (DocumentResult, CodeResult)) and r.chunk_text:
            text_hash = normalize_content_for_dedup(r.chunk_text)
            if text_hash in seen_hashes:
                primary = seen_hashes[text_hash]
                primary.duplicate_count += 1
                if r.file_name not in primary.duplicate_files:
                    primary.duplicate_files.append(r.file_name)
                continue
            else:
                seen_hashes[text_hash] = r
                collapsed.append(r)
        else:
            collapsed.append(r)

    return collapsed


class SearchService:
    def __init__(self, db: Session | None = None) -> None:
        self._vector_service = VectorService()
        self._text_embedder = TextEmbeddingService()
        self._image_embedder = ImageEmbeddingService()
        self._settings = get_settings()
        self._db = db

    def search(
        self,
        query: str,
        top_k: int | None = None,
        file_type: FileType | None = None,
    ) -> SearchResponse:
        k = top_k or 200

        intent = detect_intent(query)
        wants = file_type if file_type is not None else intent.file_type
        search_query = intent.query if file_type is None else query

        searching_images = (wants == FileType.image)
        is_exact = is_exact_term_query(search_query)

        # -------------------------------------------------------------
        # 1. IMAGE SEARCH (Strict visual modality separation)
        # -------------------------------------------------------------
        if searching_images:
            db = self._db or SessionLocal()
            should_close = (self._db is None)
            try:
                meta_service = MetadataService(db)
                filename_matches = meta_service.search_by_filename(search_query, file_type=FileType.image)
            finally:
                if should_close:
                    db.close()

            image_results: list[ImageResult] = []
            seen_img_paths: set[str] = set()

            for file_meta, score in filename_matches:
                if Path(file_meta.path).is_file() and file_meta.path not in seen_img_paths:
                    seen_img_paths.add(file_meta.path)
                    image_results.append(
                        ImageResult(
                            file_id=file_meta.id,
                            file_name=file_meta.file_name,
                            path=file_meta.path,
                            similarity=score,
                            width=file_meta.image_width or 0,
                            height=file_meta.image_height or 0,
                            match_source="lexical",
                        )
                    )

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
                if img_res.path not in seen_img_paths:
                    seen_img_paths.add(img_res.path)
                    image_results.append(img_res)

            image_results.sort(key=lambda r: r.similarity, reverse=True)
            return SearchResponse(query=query, results=image_results[:k], filtered_to=wants)

        # -------------------------------------------------------------
        # 2. DOCUMENT & CODE SEARCH (Hybrid Lexical + Semantic Retrieval)
        # -------------------------------------------------------------
        db = self._db or SessionLocal()
        should_close = (self._db is None)

        lexical_hits_by_path: dict[str, SearchResult] = {}
        lexical_scores: dict[str, float] = {}

        try:
            meta_service = MetadataService(db)

            # Leg A1: Filename Matches
            fn_matches = meta_service.search_by_filename(search_query, file_type=wants)
            for file_meta, fn_score in fn_matches:
                if not Path(file_meta.path).is_file():
                    continue
                p = file_meta.path
                lexical_scores[p] = max(lexical_scores.get(p, 0.0), fn_score)
                if file_meta.file_type == FileType.document.value:
                    lexical_hits_by_path[p] = DocumentResult(
                        file_id=file_meta.id,
                        file_name=file_meta.file_name,
                        path=file_meta.path,
                        similarity=fn_score,
                        page_number=1,
                        chunk_text=f"File: {file_meta.file_name}",
                        chunk_index=0,
                        match_source="lexical",
                    )
                elif file_meta.file_type == FileType.code.value:
                    lexical_hits_by_path[p] = CodeResult(
                        file_id=file_meta.id,
                        file_name=file_meta.file_name,
                        path=file_meta.path,
                        similarity=fn_score,
                        language=file_meta.language or "code",
                        symbol=None,
                        line_start=1,
                        line_end=1,
                        chunk_text=f"File: {file_meta.file_name}",
                        chunk_index=0,
                        match_source="lexical",
                    )

            # Leg A2: Full-Text Chunk Matches (BM25 / Keyword Overlap)
            chunk_matches = meta_service.search_chunks_lexical(search_query, file_type=wants, limit=k * 2)
            for chunk_rec, chunk_score in chunk_matches:
                if not Path(chunk_rec.path).is_file():
                    continue
                p = chunk_rec.path
                lexical_scores[p] = max(lexical_scores.get(p, 0.0), chunk_score)
                if chunk_rec.file_type == FileType.document.value:
                    lexical_hits_by_path[p] = DocumentResult(
                        file_id=chunk_rec.file_id,
                        file_name=chunk_rec.file_name,
                        path=chunk_rec.path,
                        similarity=chunk_score,
                        page_number=chunk_rec.page_number,
                        chunk_text=chunk_rec.chunk_text,
                        chunk_index=chunk_rec.chunk_index,
                        match_source="lexical",
                    )
                elif chunk_rec.file_type == FileType.code.value:
                    lexical_hits_by_path[p] = CodeResult(
                        file_id=chunk_rec.file_id,
                        file_name=chunk_rec.file_name,
                        path=chunk_rec.path,
                        similarity=chunk_score,
                        language=chunk_rec.language or "code",
                        symbol=chunk_rec.symbol,
                        line_start=chunk_rec.line_start or 1,
                        line_end=chunk_rec.line_end or 1,
                        chunk_text=chunk_rec.chunk_text,
                        chunk_index=chunk_rec.chunk_index,
                        match_source="lexical",
                    )
        finally:
            if should_close:
                db.close()

        # Leg B: Semantic Vector Search
        query_vector = self._text_embedder.encode_query(search_query)
        semantic_hits_by_path: dict[str, SearchResult] = {}
        semantic_scores: dict[str, float] = {}

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
                drop_below_floor=True,
            ):
                doc_res = self._to_document_result(hit)
                p = doc_res.path
                if p not in semantic_hits_by_path or doc_res.similarity > semantic_scores.get(p, 0.0):
                    semantic_hits_by_path[p] = doc_res
                    semantic_scores[p] = doc_res.similarity

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
                p = code_res.path
                if p not in semantic_hits_by_path or code_res.similarity > semantic_scores.get(p, 0.0):
                    semantic_hits_by_path[p] = code_res
                    semantic_scores[p] = code_res.similarity

        # -------------------------------------------------------------
        # 3. EXACT-TERM GROUNDING & ZERO-MATCH CHECK
        # -------------------------------------------------------------
        # If the user queried an exact acronym/term (e.g. AWS) and there are ZERO lexical matches:
        # Require exceptionally high semantic confidence (> 0.85) to prevent hallucinations/noise.
        if is_exact:
            if not lexical_hits_by_path:
                max_sem = max(semantic_scores.values()) if semantic_scores else 0.0
                if max_sem < 0.85:
                    # Grounded zero-match: do not return noise documents for un-matched exact terms
                    return SearchResponse(query=query, results=[], filtered_to=wants)

        # -------------------------------------------------------------
        # 4. RECIPROCAL RANK FUSION (RRF)
        # -------------------------------------------------------------
        all_candidate_paths = set(lexical_hits_by_path.keys()) | set(semantic_hits_by_path.keys())

        # Sort lexical and semantic ranked lists
        lexical_sorted = sorted(lexical_hits_by_path.keys(), key=lambda p: lexical_scores.get(p, 0.0), reverse=True)
        semantic_sorted = sorted(semantic_hits_by_path.keys(), key=lambda p: semantic_scores.get(p, 0.0), reverse=True)

        lexical_rank_map = {p: i + 1 for i, p in enumerate(lexical_sorted)}
        semantic_rank_map = {p: i + 1 for i, p in enumerate(semantic_sorted)}

        RRF_K = 60
        w_lex = 0.85 if is_exact else 0.40
        w_sem = 0.15 if is_exact else 0.60

        fused_results: list[tuple[SearchResult, float]] = []

        for p in all_candidate_paths:
            r_lex = lexical_rank_map.get(p)
            r_sem = semantic_rank_map.get(p)

            score_lex = (w_lex / (RRF_K + r_lex)) if r_lex else 0.0
            score_sem = (w_sem / (RRF_K + r_sem)) if r_sem else 0.0
            rrf_score = score_lex + score_sem

            # Determine best card representation and match source
            if p in lexical_hits_by_path and p in semantic_hits_by_path:
                card = semantic_hits_by_path[p]
                card.match_source = "hybrid"
                card.similarity = min(1.0, (lexical_scores.get(p, 0.0) * 0.5) + (semantic_scores.get(p, 0.0) * 0.5))
            elif p in lexical_hits_by_path:
                card = lexical_hits_by_path[p]
                card.match_source = "lexical"
                card.similarity = lexical_scores.get(p, 1.0)
            else:
                card = semantic_hits_by_path[p]
                card.match_source = "semantic"
                card.similarity = semantic_scores.get(p, 0.0)

            fused_results.append((card, rrf_score))

        # Sort by RRF score descending
        fused_results.sort(key=lambda item: item[1], reverse=True)
        ordered_candidates = [item[0] for item in fused_results]

        # -------------------------------------------------------------
        # 5. DEDUPLICATION (Per-File & Cross-File Near Duplicates)
        # -------------------------------------------------------------
        deduped_per_file = best_per_file(ordered_candidates)
        collapsed = collapse_near_duplicates(deduped_per_file)

        return SearchResponse(query=query, results=collapsed[:k], filtered_to=wants)

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
            match_source="semantic",
        )

    @staticmethod
    def _to_code_result(hit: SearchHit) -> CodeResult:
        p = hit.payload
        return CodeResult(
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            similarity=hit.score,
            language=p.get("language", "code"),
            symbol=p.get("symbol"),
            line_start=p.get("line_start", 1),
            line_end=p.get("line_end", 1),
            chunk_text=p["chunk_text"],
            chunk_index=p["chunk_index"],
            match_source="semantic",
        )

    @staticmethod
    def _to_image_result(hit: SearchHit) -> ImageResult:
        p = hit.payload
        return ImageResult(
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            similarity=hit.score,
            width=p.get("width", 0),
            height=p.get("height", 0),
            match_source="semantic",
        )
