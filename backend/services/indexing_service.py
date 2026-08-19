import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.chunking.code_chunker import build_embedding_text, chunk_code
from core.chunking.text_chunker import chunk_pages
from core.embeddings.image_embedding_service import ImageEmbeddingService
from core.embeddings.text_embedding_service import TextEmbeddingService
from core.parsers.code_parser import parse_code
from core.parsers.document_parser import parse_document
from core.parsers.image_reader import read_image_info
from core.scanner.folder_scanner import FolderScanner, ScannedFile
from core.vectorstore.qdrant_client import VectorService
from models.schemas.file_schemas import FileType
from services.metadata_service import MetadataService
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexProgress:
    processed: int
    total: int
    current_file: str


@dataclass(frozen=True)
class IndexSummary:
    indexed: int
    removed: int


ProgressCallback = Callable[[IndexProgress], None]


class IndexingService:
    def __init__(self, metadata_service: MetadataService) -> None:
        self._metadata_service = metadata_service
        self._vector_service = VectorService()
        self._text_embedder = TextEmbeddingService()
        self._image_embedder = ImageEmbeddingService()
        self._vector_service.ensure_collection()

    def index_folder(self, root: str, on_progress: ProgressCallback | None = None) -> IndexSummary:
        root_path = Path(root).resolve()
        files = list(FolderScanner().scan(root))
        total = len(files)
        indexed_count = 0

        for i, scanned in enumerate(files, start=1):
            try:
                self._index_file(scanned, root)
                indexed_count += 1
            except Exception:
                logger.exception("Failed to index file: %s", scanned.path)

            if on_progress:
                on_progress(IndexProgress(processed=i, total=total, current_file=scanned.path))

        removed_count = self._remove_deleted_files(root_path, {f.path for f in files})
        return IndexSummary(indexed=indexed_count, removed=removed_count)

    def _remove_deleted_files(self, root: Path, seen_paths: set[str]) -> int:
        """Drop index entries for files that have since been deleted from disk.

        A scan only ever adds what it finds, so without this step a file the
        user deleted keeps its row in SQLite and its vectors in Qdrant
        forever: it still ranks in search, and "Open" then fails with a file
        not found. Re-scanning a folder has to mean "make the index match the
        folder", not "add whatever is there now".

        Two guards keep this from deleting more than it should. Only records
        *under the folder being scanned* are candidates, so indexing folder A
        never prunes what was indexed from folder B. And a candidate is only
        removed once `exists()` confirms it is really gone — a file the
        scanner merely skipped this run (an unreadable directory, a source
        file that grew past `code_max_file_bytes`, a newly ignored directory)
        is still on disk and keeps its entry.
        """
        seen = {os.path.normcase(path) for path in seen_paths}
        stale = [
            path
            for path in self._metadata_service.list_paths()
            if os.path.normcase(path) not in seen
            and Path(path).is_relative_to(root)
            and not Path(path).exists()
        ]

        for path in stale:
            logger.info("Removing deleted file from index: %s", path)
            self._vector_service.delete_by_path(path)

        return self._metadata_service.delete_paths(stale)

    def _index_file(self, scanned: ScannedFile, root: str) -> None:
        # Clear any vectors from a previous indexing run of this same path
        # first, so re-scanning never produces duplicate chunks.
        self._vector_service.delete_by_path(scanned.path)

        registered = self._metadata_service.upsert_file(
            file_name=scanned.file_name,
            file_type=scanned.file_type,
            extension=scanned.extension,
            path=scanned.path,
            size_bytes=scanned.size_bytes,
        )

        if scanned.file_type == FileType.document:
            self._index_document(scanned, registered.id)
        elif scanned.file_type == FileType.code:
            self._index_code(scanned, registered.id, root)
        else:
            self._index_image(scanned, registered.id)

    def _index_document(self, scanned: ScannedFile, file_id: str) -> None:
        pages = parse_document(scanned.path, scanned.extension)
        chunks = chunk_pages(pages)

        if chunks:
            vectors = self._text_embedder.encode_documents([c.chunk_text for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                self._vector_service.upsert_text_chunk(
                    file_id=file_id,
                    file_name=scanned.file_name,
                    path=scanned.path,
                    page_number=chunk.page_number,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    vector=vector,
                )
        else:
            logger.warning("No extractable text in %s", scanned.path)

        self._metadata_service.upsert_file(
            file_name=scanned.file_name,
            file_type=FileType.document,
            extension=scanned.extension,
            path=scanned.path,
            size_bytes=scanned.size_bytes,
            chunk_count=len(chunks),
        )

    def _index_code(self, scanned: ScannedFile, file_id: str, root: str) -> None:
        parsed = parse_code(scanned.path, scanned.extension)
        chunks = chunk_code(parsed.text, parsed.language)

        if chunks:
            # The path relative to the folder the user indexed — "services/
            # rag_service.py" rather than the machine-specific absolute path,
            # which would put this user's home directory into every vector.
            try:
                relative_path = Path(scanned.path).relative_to(Path(root).resolve()).as_posix()
            except ValueError:
                relative_path = scanned.file_name

            vectors = self._text_embedder.encode_documents(
                [
                    build_embedding_text(c, relative_path=relative_path, language=parsed.language)
                    for c in chunks
                ]
            )
            for chunk, vector in zip(chunks, vectors):
                self._vector_service.upsert_code_chunk(
                    file_id=file_id,
                    file_name=scanned.file_name,
                    path=scanned.path,
                    language=parsed.language,
                    symbol=chunk.symbol,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    vector=vector,
                )
        else:
            logger.info("No indexable code in %s", scanned.path)

        self._metadata_service.upsert_file(
            file_name=scanned.file_name,
            file_type=FileType.code,
            extension=scanned.extension,
            path=scanned.path,
            size_bytes=scanned.size_bytes,
            chunk_count=len(chunks),
            language=parsed.language,
        )

    def _index_image(self, scanned: ScannedFile, file_id: str) -> None:
        info = read_image_info(scanned.path)
        vector = self._image_embedder.encode_image(scanned.path)
        self._vector_service.upsert_image(
            file_id=file_id,
            file_name=scanned.file_name,
            path=scanned.path,
            width=info.width,
            height=info.height,
            vector=vector,
        )

        self._metadata_service.upsert_file(
            file_name=scanned.file_name,
            file_type=FileType.image,
            extension=scanned.extension,
            path=scanned.path,
            size_bytes=scanned.size_bytes,
            image_width=info.width,
            image_height=info.height,
        )
