import difflib
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from core.query.literal import LiteralTerms, StemIndex
from database.models import ChunkRecord, FileRecord, WatchedFolder
from models.schemas.file_schemas import FileMetadata, FileType, IndexStats
from models.schemas.index_schemas import WatchedFolderResponse
from utils.logger import get_logger

logger = get_logger(__name__)

_QUESTION_WORDS = {
    "how", "what", "why", "when", "where", "who", "which",
    "can", "could", "is", "are", "do", "does", "did", "was", "were",
}
_COMMON_STOPWORDS = {
    "how", "much", "was", "the", "what", "why", "when", "where", "who", "which",
    "for", "with", "and", "from", "any", "some", "all", "our", "you", "your",
    "this", "that", "these", "those", "have", "has", "had", "about",
}


@dataclass(frozen=True)
class LiteralFileMatch:
    """One file that literally contains what the query asked for."""

    file: FileMetadata
    name_hit: bool
    """The demand is met by the file name alone - what puts it in the top tier."""
    content_hit: bool
    """Some passage inside the file carries the query's words."""
    content_score: float
    """How completely, and how repeatedly, the contents answer the query."""
    best_chunk: ChunkRecord | None
    """The passage to show on the card: the one covering most of the query."""


@dataclass(frozen=True)
class LiteralMatches:
    by_path: dict[str, LiteralFileMatch]
    ignored_terms: tuple[str, ...]
    """Query words that appear in no indexed file, so could not be demanded."""
    any_term_findable: bool
    """False when nothing in the index carries any word of the query."""


def normalize_content_for_dedup(text: str) -> str:
    """Produces a deterministic MD5 hash of alphanumeric lowercased tokens."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", text.lower())
    return hashlib.md5(cleaned[:4000].encode("utf-8", errors="ignore")).hexdigest()


def calculate_filename_match_score(query_str: str, file_name: str) -> float:
    """Calculates a normalized score (0.0 to 1.0) indicating how well a filename matches a search query."""
    fn_lower = file_name.lower()
    q_lower = query_str.strip().lower()

    if not q_lower:
        return 0.0

    is_question = bool(
        any(q_lower.startswith(f"{qw} ") for qw in _QUESTION_WORDS) or q_lower.endswith("?")
    )

    if not is_question and q_lower in fn_lower:
        return 1.0

    q_tokens = [
        t for t in re.findall(r"[a-z0-9]+", q_lower)
        if len(t) >= 2 and t not in _COMMON_STOPWORDS
    ]

    if not q_tokens:
        return 0.0

    if is_question:
        if len(q_tokens) >= 2:
            matched_q_tokens = sum(1 for t in q_tokens if t in fn_lower)
            if matched_q_tokens == len(q_tokens):
                return 1.0
        return 0.0

    fn_tokens = re.findall(r"[a-z0-9]+", fn_lower)

    matched_tokens = 0
    for q_tok in q_tokens:
        if q_tok in fn_lower:
            matched_tokens += 1
        else:
            close = [
                t
                for t in fn_tokens
                if difflib.SequenceMatcher(None, q_tok, t).ratio() >= 0.75
            ]
            if close:
                matched_tokens += 1

    if matched_tokens == len(q_tokens):
        return 1.0

    return 0.0


class MetadataService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_file(
        self,
        *,
        file_name: str,
        file_type: FileType,
        extension: str,
        path: str,
        size_bytes: int,
        chunk_count: int | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        language: str | None = None,
    ) -> FileMetadata:
        resolved = str(Path(path).resolve())
        record = self.db.scalar(select(FileRecord).where(FileRecord.path == resolved))
        if record is None:
            record = FileRecord(
                file_name=file_name,
                file_type=file_type.value,
                extension=extension,
                path=resolved,
                size_bytes=size_bytes,
            )
            self.db.add(record)
            logger.info("Registering new file: %s", resolved)
        else:
            record.file_name = file_name
            record.file_type = file_type.value
            record.extension = extension
            record.size_bytes = size_bytes
            logger.info("Re-indexing existing file: %s", resolved)

        record.chunk_count = chunk_count
        record.image_width = image_width
        record.image_height = image_height
        record.language = language

        self.db.commit()
        self.db.refresh(record)
        return FileMetadata.model_validate(record)

    def upsert_chunks(
        self,
        *,
        file_id: str,
        file_name: str,
        file_type: FileType,
        path: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        resolved = str(Path(path).resolve())
        # Delete existing chunks for this path
        self.db.execute(delete(ChunkRecord).where(ChunkRecord.path == resolved))

        for c in chunks:
            chunk_text = c.get("chunk_text", "")
            rec = ChunkRecord(
                file_id=file_id,
                file_name=file_name,
                file_type=file_type.value,
                path=resolved,
                chunk_index=c.get("chunk_index", 0),
                page_number=c.get("page_number"),
                line_start=c.get("line_start"),
                line_end=c.get("line_end"),
                symbol=c.get("symbol"),
                language=c.get("language"),
                chunk_text=chunk_text,
                content_hash=normalize_content_for_dedup(chunk_text),
            )
            self.db.add(rec)
        self.db.commit()

    def delete_chunks_by_path(self, path: str) -> None:
        resolved = str(Path(path).resolve())
        self.db.execute(delete(ChunkRecord).where(ChunkRecord.path == resolved))
        self.db.commit()

    def search_chunks_lexical(
        self, query: str, file_type: FileType | None = None, limit: int = 200
    ) -> list[tuple[ChunkRecord, float]]:
        """Performs lexical full-text matching against indexed chunks."""
        q_clean = query.strip()
        if not q_clean:
            return []

        # Tokenize query
        q_tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", q_clean) if len(t) >= 1]
        if not q_tokens:
            return []

        stmt = select(ChunkRecord)
        if file_type is not None:
            stmt = stmt.where(ChunkRecord.file_type == file_type.value)

        # Require at least one token match in chunk_text
        try:
            records = self.db.scalars(stmt).all()
        except OperationalError:
            return []

        matches: list[tuple[ChunkRecord, float]] = []
        is_exact_phrase = q_clean.startswith('"') and q_clean.endswith('"')
        target_phrase = q_clean.strip('"').lower()

        for rec in records:
            text_lower = rec.chunk_text.lower()

            if is_exact_phrase:
                if target_phrase in text_lower:
                    matches.append((rec, 1.0))
                continue

            # Check if all or subset of tokens match
            matched_count = sum(1 for tok in q_tokens if tok in text_lower)
            if matched_count > 0:
                fraction = matched_count / len(q_tokens)
                # Exact single word / acronym match bonus
                if len(q_tokens) == 1 and q_tokens[0] in text_lower:
                    score = 1.0
                else:
                    score = fraction * 0.95
                if score >= 0.5:
                    matches.append((rec, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]

    def search_files_literal(
        self, terms: LiteralTerms, file_type: FileType | None = None
    ) -> LiteralMatches:
        """Find every file that literally contains what the query asked for.

        The unit of judgement is the *file*, not the chunk: a name written in a
        cover page and a surname repeated in the body are one match, even
        though no single passage holds both. Chunking is an implementation
        detail of retrieval and the user never agreed to it.
        """
        if terms.is_empty:
            return LiteralMatches(by_path={}, ignored_terms=(), any_term_findable=True)

        try:
            files = self.list_files(file_type=file_type)
            chunk_stmt = select(ChunkRecord)
            if file_type is not None:
                chunk_stmt = chunk_stmt.where(ChunkRecord.file_type == file_type.value)
            chunks = self.db.scalars(chunk_stmt).all()
        except OperationalError:
            return LiteralMatches(by_path={}, ignored_terms=(), any_term_findable=False)

        chunks_by_path: dict[str, list[ChunkRecord]] = {}
        for chunk in chunks:
            chunks_by_path.setdefault(chunk.path, []).append(chunk)

        stems = terms.stems
        phrase = terms.phrase

        # Pass 1: index every file, and learn which query words exist at all.
        indexed: list[dict[str, Any]] = []
        findable: set[str] = set()

        for file_meta in files:
            name_index = StemIndex()
            name_index.add(file_meta.file_name)

            content_index = StemIndex()
            best_chunk: ChunkRecord | None = None
            best_covered = -1
            matching_chunks = 0
            phrase_in_content = False

            for chunk in sorted(chunks_by_path.get(file_meta.path, []), key=lambda c: c.chunk_index):
                chunk_index = StemIndex()
                chunk_index.add(chunk.chunk_text)
                content_index.merge(chunk_index)

                covered = sum(1 for s in stems if chunk_index.contains(s))
                if phrase and phrase in " ".join(chunk.chunk_text.lower().split()):
                    # A quoted phrase found whole is the most complete answer a
                    # passage can give, whatever its individual words scored.
                    phrase_in_content = True
                    covered = max(covered, len(stems), 1)
                if covered:
                    matching_chunks += 1
                # The passage shown on the card is the one that answers most of
                # the query, and the earliest such passage when they tie.
                if covered > best_covered:
                    best_covered = covered
                    best_chunk = chunk

            demanded = len(stems) or 1
            best_coverage = min(1.0, max(best_covered, 0) / demanded)

            phrase_in_name = bool(phrase) and phrase in " ".join(file_meta.file_name.lower().split())
            in_name = {s for s in stems if name_index.contains(s)}
            in_content = {s for s in stems if content_index.contains(s)}
            findable |= in_name | in_content

            indexed.append({
                "file": file_meta,
                "in_name": in_name,
                "in_content": in_content,
                "phrase_in_name": phrase_in_name,
                "phrase_in_content": phrase_in_content,
                "best_chunk": best_chunk,
                "best_coverage": best_coverage,
                "matching_chunks": matching_chunks,
            })

        # A word the index has never seen cannot be demanded of any file - see
        # this module's docstring. A *quoted* phrase is exempt: quoting is the
        # user saying "this exactly, or nothing".
        required = {s for s in stems if s in findable}
        ignored = tuple(
            token for token, s in zip(terms.tokens, stems) if s not in findable
        )
        if not required and not phrase:
            return LiteralMatches(by_path={}, ignored_terms=ignored, any_term_findable=False)

        by_path: dict[str, LiteralFileMatch] = {}
        for entry in indexed:
            covered_anywhere = entry["in_name"] | entry["in_content"]
            if not required.issubset(covered_anywhere):
                continue
            if phrase and not (entry["phrase_in_name"] or entry["phrase_in_content"]):
                continue

            name_hit = required.issubset(entry["in_name"]) and (not phrase or entry["phrase_in_name"])
            content_hit = bool(entry["in_content"] & required) or entry["phrase_in_content"]

            # How completely one passage answers the query, plus a small credit
            # for the file returning to the subject rather than mentioning it once.
            content_score = min(
                1.0,
                0.7 * entry["best_coverage"] + 0.3 * min(1.0, entry["matching_chunks"] / 3.0),
            )

            file_meta = entry["file"]
            by_path[file_meta.path] = LiteralFileMatch(
                file=file_meta,
                name_hit=name_hit,
                content_hit=content_hit,
                content_score=content_score,
                best_chunk=entry["best_chunk"] if content_hit else None,
            )

        return LiteralMatches(by_path=by_path, ignored_terms=ignored, any_term_findable=True)

    def get_by_path(self, path: str) -> FileMetadata | None:
        resolved = str(Path(path).resolve())
        record = self.db.scalar(select(FileRecord).where(FileRecord.path == resolved))
        return FileMetadata.model_validate(record) if record else None

    def get_by_id(self, file_id: str) -> FileMetadata | None:
        record = self.db.get(FileRecord, file_id)
        return FileMetadata.model_validate(record) if record else None

    def list_files(self, file_type: FileType | None = None) -> list[FileMetadata]:
        stmt = select(FileRecord)
        if file_type is not None:
            stmt = stmt.where(FileRecord.file_type == file_type.value)
        records = self.db.scalars(stmt).all()
        return [FileMetadata.model_validate(r) for r in records]

    def search_by_filename(
        self, query: str, file_type: FileType | None = None
    ) -> list[tuple[FileMetadata, float]]:
        """Search metadata records where the file name or language matches the query."""
        try:
            all_files = self.list_files(file_type=file_type)
        except OperationalError:
            return []

        q_clean = query.strip().lower()
        scored: list[tuple[FileMetadata, float]] = []

        for file_meta in all_files:
            score = calculate_filename_match_score(query, file_meta.file_name)

            if score < 0.80:
                if q_clean in ("python", "py") and (
                    file_meta.extension in (".py", ".pyw", ".ipynb")
                    or (file_meta.language and file_meta.language.lower() == "python")
                ):
                    score = 1.0
                elif q_clean in ("typescript", "ts") and file_meta.extension in (".ts", ".tsx"):
                    score = 1.0
                elif q_clean in ("javascript", "js") and file_meta.extension in (".js", ".jsx"):
                    score = 1.0

            if score >= 0.80:
                scored.append((file_meta, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def list_paths(self) -> list[str]:
        return list(self.db.scalars(select(FileRecord.path)).all())

    def delete_by_path(self, path: str) -> bool:
        resolved = str(Path(path).resolve())
        self.delete_chunks_by_path(resolved)
        record = self.db.scalar(select(FileRecord).where(FileRecord.path == resolved))
        if record is None:
            return False
        self.db.delete(record)
        self.db.commit()
        return True

    def delete_paths(self, paths: Sequence[str]) -> int:
        if not paths:
            return 0
        resolved_paths = [str(Path(p).resolve()) for p in paths]
        for p in resolved_paths:
            self.delete_chunks_by_path(p)
        records = self.db.scalars(select(FileRecord).where(FileRecord.path.in_(resolved_paths))).all()
        for record in records:
            self.db.delete(record)
        self.db.commit()
        return len(records)

    def get_stats(self) -> IndexStats:
        total_files = self.db.scalar(select(func.count(FileRecord.id))) or 0
        total_documents = (
            self.db.scalar(
                select(func.count(FileRecord.id)).where(FileRecord.file_type == FileType.document.value)
            )
            or 0
        )
        total_images = (
            self.db.scalar(
                select(func.count(FileRecord.id)).where(FileRecord.file_type == FileType.image.value)
            )
            or 0
        )
        total_code = (
            self.db.scalar(
                select(func.count(FileRecord.id)).where(FileRecord.file_type == FileType.code.value)
            )
            or 0
        )
        total_chunks = self.db.scalar(select(func.coalesce(func.sum(FileRecord.chunk_count), 0))) or 0
        total_size_bytes = self.db.scalar(select(func.coalesce(func.sum(FileRecord.size_bytes), 0))) or 0

        return IndexStats(
            total_files=total_files,
            total_documents=total_documents,
            total_images=total_images,
            total_code=total_code,
            total_chunks=total_chunks,
            total_size_bytes=total_size_bytes,
        )

    # --- Watched Folder Operations ---

    def add_watched_folder(self, path: str) -> WatchedFolderResponse:
        resolved = str(Path(path).resolve())
        folder = self.db.scalar(select(WatchedFolder).where(WatchedFolder.path == resolved))
        if folder is None:
            folder = WatchedFolder(path=resolved, is_active=True)
            self.db.add(folder)
            logger.info("Added watched folder: %s", resolved)
        else:
            folder.is_active = True
            logger.info("Re-activated watched folder: %s", resolved)
        self.db.commit()
        self.db.refresh(folder)
        return WatchedFolderResponse.model_validate(folder)

    def remove_watched_folder(self, path: str) -> bool:
        resolved = str(Path(path).resolve())
        folder = self.db.scalar(select(WatchedFolder).where(WatchedFolder.path == resolved))
        if folder is None:
            return False
        self.db.delete(folder)
        self.db.commit()
        logger.info("Removed watched folder: %s", resolved)
        return True

    def list_watched_folders(self, only_active: bool = True) -> list[WatchedFolderResponse]:
        stmt = select(WatchedFolder)
        if only_active:
            stmt = stmt.where(WatchedFolder.is_active.is_(True))
        folders = self.db.scalars(stmt).all()
        return [WatchedFolderResponse.model_validate(f) for f in folders]

    def update_watched_folder_scanned(self, path: str) -> None:
        resolved = str(Path(path).resolve())
        folder = self.db.scalar(select(WatchedFolder).where(WatchedFolder.path == resolved))
        if folder is not None:
            folder.last_scanned_at = datetime.now(timezone.utc)
            self.db.commit()
