import re
from pathlib import Path
from typing import Sequence

from sqlalchemy.orm import Session

from core.embeddings.image_embedding_service import ImageEmbeddingService
from core.embeddings.text_embedding_service import TextEmbeddingService
from core.query.intent import detect_intent
from core.query.literal import LiteralTerms, extract_literal_terms
from core.vectorstore.qdrant_client import SearchHit, VectorService
from database.models import ChunkRecord
from database.session import SessionLocal
from models.schemas.file_schemas import FileMetadata, FileType
from models.schemas.search_schemas import (
    CodeResult,
    DocumentResult,
    ImageResult,
    SearchResponse,
    SearchResult,
)
from services.metadata_service import (
    LiteralMatches,
    MetadataService,
    normalize_content_for_dedup,
)
from utils.config import get_settings

# Which tier a file lands in. Rank is decided by *why* a file matched before it
# is decided by how strongly: a file named for what you searched is the answer
# you meant, even when a passage buried in another file scores higher.
TIER_NAME = 0
"""The file name carries the query."""
TIER_CONTENT = 1
"""The query's words are inside the file."""
TIER_SEMANTIC = 2
"""Neither - only meaning is close. Never reached while strict matching is on."""


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
        """Find files, name matches first, then files containing the words.

        With `search_strict_lexical` on - the default - a file that does not
        contain what was typed is never returned, however close its embedding
        sits. Semantic similarity orders the survivors and picks which passage
        to show; it cannot put a file on screen by itself. Searching a name or
        a college is a containment question, and a confidently-ranked file that
        does not contain the name is simply a wrong answer.
        """
        k = top_k or 200

        intent = detect_intent(query)
        wants = file_type if file_type is not None else intent.file_type
        search_query = intent.query if file_type is None else query

        terms = extract_literal_terms(search_query)
        # "images" on its own demands nothing literal - it names a type and has
        # no subject, so there is no word to require and the gate stays open.
        strict = self._settings.search_strict_lexical and not terms.is_empty

        if wants == FileType.image:
            return self._search_images(query, search_query, k, wants, strict)

        return self._search_text(query, search_query, terms, strict, k, wants)

    # ------------------------------------------------------------------
    # documents and code
    # ------------------------------------------------------------------

    def _search_text(
        self,
        query: str,
        search_query: str,
        terms: LiteralTerms,
        strict: bool,
        k: int,
        wants: FileType | None,
    ) -> SearchResponse:
        db = self._db or SessionLocal()
        should_close = self._db is None
        try:
            meta_service = MetadataService(db)
            literal: LiteralMatches = meta_service.search_files_literal(terms, file_type=wants)
            name_matches = {
                file_meta.path: (file_meta, score)
                for file_meta, score in meta_service.search_by_filename(search_query, file_type=wants)
                if Path(file_meta.path).is_file()
            }
        finally:
            if should_close:
                db.close()

        semantic_cards, semantic_scores = self._semantic_text_pass(search_query, k, wants)

        # The gate. Everything below only orders what this admits.
        admitted: set[str] = set(name_matches)
        admitted |= {
            path for path, match in literal.by_path.items() if Path(path).is_file()
        }
        if not strict:
            admitted |= set(semantic_cards)

        is_exact = is_exact_term_query(search_query)
        w_lex, w_sem = (0.85, 0.15) if is_exact else (0.55, 0.45)

        ranked: list[tuple[int, float, SearchResult]] = []
        for path in admitted:
            match = literal.by_path.get(path)
            named = name_matches.get(path)
            semantic = semantic_scores.get(path, 0.0)

            name_hit = named is not None or (match is not None and match.name_hit)
            content_hit = match is not None and match.content_hit

            if name_hit:
                tier = TIER_NAME
            elif match is not None:
                tier = TIER_CONTENT
            else:
                tier = TIER_SEMANTIC

            lexical = 0.0
            if named is not None:
                lexical = max(lexical, named[1])
            if match is not None:
                lexical = max(lexical, match.content_score)

            strength = min(1.0, w_lex * lexical + w_sem * semantic)
            if tier == TIER_SEMANTIC:
                strength = semantic

            # A file named for the query is the strongest evidence there is, so
            # it never *reads* as a weak hit - but the floor is applied to the
            # number on the card, not to the number it is ranked by, or every
            # name match would tie and their order would be arbitrary.
            similarity = max(strength, 0.85) if tier == TIER_NAME else strength

            card = self._card_for(
                path,
                chunk=match.best_chunk if match else None,
                file_meta=(named[0] if named else (match.file if match else None)),
                semantic_card=semantic_cards.get(path),
            )
            if card is None:
                continue

            card.similarity = similarity
            card.match_source = (
                "hybrid" if (tier != TIER_SEMANTIC and semantic > 0.0)
                else ("lexical" if tier != TIER_SEMANTIC else "semantic")
            )
            ranked.append((tier, strength, card))

        ranked.sort(key=lambda item: (item[0], -item[1], item[2].file_name))
        ordered = [card for _, _, card in ranked]

        collapsed = collapse_near_duplicates(best_per_file(ordered))
        return SearchResponse(
            query=query,
            results=collapsed[:k],
            filtered_to=wants,
            ignored_terms=list(literal.ignored_terms) if strict else [],
        )

    def _semantic_text_pass(
        self, search_query: str, k: int, wants: FileType | None
    ) -> tuple[dict[str, SearchResult], dict[str, float]]:
        """Rank by meaning. Used to order and illustrate, never to admit."""
        query_vector = self._text_embedder.encode_query(search_query)
        cards: dict[str, SearchResult] = {}
        scores: dict[str, float] = {}

        partitions = (
            (FileType.document, self._to_document_result,
             self._settings.search_text_score_floor, self._settings.search_text_score_ceil),
            (FileType.code, self._to_code_result,
             self._settings.search_code_score_floor, self._settings.search_code_score_ceil),
        )
        for partition, to_result, floor, ceil in partitions:
            if wants not in (None, partition):
                continue
            hits = drop_missing_files(
                self._vector_service.search_text(
                    query_vector, top_k=k * 4, file_type=partition.value
                )
            )
            for hit in _calibrate(hits, floor=floor, ceil=ceil, drop_below_floor=True):
                result = to_result(hit)
                if result.similarity > scores.get(result.path, 0.0) or result.path not in scores:
                    cards[result.path] = result
                    scores[result.path] = result.similarity
        return cards, scores

    @staticmethod
    def _card_for(
        path: str,
        *,
        chunk: ChunkRecord | None,
        file_meta: FileMetadata | None,
        semantic_card: SearchResult | None,
    ) -> SearchResult | None:
        """Build the result card, preferring the passage that holds the words.

        A card showing a passage the query does not appear in is how a strict
        search still looks wrong to the person reading it, so the literal chunk
        wins over the semantically nearest one whenever there is one.
        """
        if chunk is not None:
            if chunk.file_type == FileType.code.value:
                return CodeResult(
                    file_id=chunk.file_id,
                    file_name=chunk.file_name,
                    path=chunk.path,
                    similarity=0.0,
                    language=chunk.language or "code",
                    symbol=chunk.symbol,
                    line_start=chunk.line_start or 1,
                    line_end=chunk.line_end or 1,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    match_source="lexical",
                )
            return DocumentResult(
                file_id=chunk.file_id,
                file_name=chunk.file_name,
                path=chunk.path,
                similarity=0.0,
                page_number=chunk.page_number,
                chunk_text=chunk.chunk_text,
                chunk_index=chunk.chunk_index,
                match_source="lexical",
            )

        if semantic_card is not None:
            return semantic_card

        if file_meta is None:
            return None

        # Matched on its name with nothing quotable inside it.
        if file_meta.file_type == FileType.code.value:
            return CodeResult(
                file_id=file_meta.id,
                file_name=file_meta.file_name,
                path=file_meta.path,
                similarity=0.0,
                language=file_meta.language or "code",
                symbol=None,
                line_start=1,
                line_end=1,
                chunk_text=f"File: {file_meta.file_name}",
                chunk_index=0,
                match_source="lexical",
            )
        if file_meta.file_type == FileType.document.value:
            return DocumentResult(
                file_id=file_meta.id,
                file_name=file_meta.file_name,
                path=file_meta.path,
                similarity=0.0,
                page_number=1,
                chunk_text=f"File: {file_meta.file_name}",
                chunk_index=0,
                match_source="lexical",
            )
        return None

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------

    def _search_images(
        self, query: str, search_query: str, k: int, wants: FileType | None, strict: bool
    ) -> SearchResponse:
        """Images rank by name first, then by what they show.

        Strict literal matching cannot apply to the contents of a photograph -
        an image holds no text to contain a word - so a CLIP match *is* the
        content tier here. Dropping it would leave image search able to find
        nothing but filenames.
        """
        db = self._db or SessionLocal()
        should_close = self._db is None
        try:
            name_matches = MetadataService(db).search_by_filename(
                search_query, file_type=FileType.image
            )
        finally:
            if should_close:
                db.close()

        ranked: list[tuple[int, float, ImageResult]] = []
        seen: set[str] = set()

        for file_meta, score in name_matches:
            if not Path(file_meta.path).is_file() or file_meta.path in seen:
                continue
            seen.add(file_meta.path)
            ranked.append((
                TIER_NAME,
                max(score, 0.85),
                ImageResult(
                    file_id=file_meta.id,
                    file_name=file_meta.file_name,
                    path=file_meta.path,
                    similarity=max(score, 0.85),
                    width=file_meta.image_width or 0,
                    height=file_meta.image_height or 0,
                    match_source="lexical",
                ),
            ))

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
            result = self._to_image_result(hit)
            if result.path in seen:
                continue
            seen.add(result.path)
            ranked.append((TIER_CONTENT, result.similarity, result))

        ranked.sort(key=lambda item: (item[0], -item[1]))
        return SearchResponse(
            query=query,
            results=[card for _, _, card in ranked][:k],
            filtered_to=wants,
        )

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
