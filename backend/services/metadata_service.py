import difflib
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from database.models import FileRecord, WatchedFolder
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


def calculate_filename_match_score(query_str: str, file_name: str) -> float:
    """Calculates a normalized score (0.0 to 1.0) indicating how well a filename matches a search query."""
    fn_lower = file_name.lower()
    q_lower = query_str.strip().lower()

    if not q_lower:
        return 0.0

    is_question = bool(
        any(q_lower.startswith(f"{qw} ") for qw in _QUESTION_WORDS) or q_lower.endswith("?")
    )

    # 1. Exact match of query string anywhere in filename (e.g. "nptel" in "nptel payment.pdf", "python" in "test_python.py")
    if not is_question and q_lower in fn_lower:
        return 1.0

    q_tokens = [
        t for t in re.findall(r"[a-z0-9]+", q_lower)
        if len(t) >= 2 and t not in _COMMON_STOPWORDS
    ]

    if not q_tokens:
        return 0.0

    if is_question:
        # For full natural language questions, only match if all non-stop query tokens match the filename
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
            # Fuzzy match token against filename tokens (e.g., 'nptl' matches 'nptel')
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

            # Special keyword handling for programming languages / file extensions
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
