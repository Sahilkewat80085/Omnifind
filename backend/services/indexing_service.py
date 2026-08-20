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
from core.scanner.folder_scanner import FolderScanner, ScannedFile, classify_extension
from core.vectorstore.qdrant_client import VectorService
from models.schemas.file_schemas import FileType
from services.metadata_service import MetadataService
from utils.config import get_settings
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

    def index_single_file(self, file_path: str, root_folder: str | None = None) -> bool:
        """Indexes or updates a single file in Qdrant and SQLite in real time."""
        path_obj = Path(file_path).resolve()
        if not path_obj.exists() or not path_obj.is_file():
            return False

        ext = path_obj.suffix.lower()
        file_type = classify_extension(ext)
        if file_type is None:
            return False

        try:
            size_bytes = path_obj.stat().st_size
        except OSError:
            return False

        if file_type == FileType.code and size_bytes > get_settings().code_max_file_bytes:
            return False

        scanned = ScannedFile(
            path=str(path_obj),
            file_name=path_obj.name,
            extension=ext,
            file_type=file_type,
            size_bytes=size_bytes,
        )

        root = root_folder or str(path_obj.parent)
        try:
            self._index_file(scanned, root)
            logger.info("Real-time auto-indexed file: %s", scanned.path)
            return True
        except Exception:
            logger.exception("Failed to auto-index file: %s", scanned.path)
            return False

    def remove_single_file(self, file_path: str) -> bool:
        """Removes a single file from Qdrant vectors and SQLite metadata in real time."""
        resolved = str(Path(file_path).resolve())
        logger.info("Real-time removing file from index: %s", resolved)
        self._vector_service.delete_by_path(resolved)
        return self._metadata_service.delete_by_path(resolved)

    def _remove_deleted_files(self, root: Path, seen_paths: set[str]) -> int:
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
            logger.warning("No extractable text in %s; indexing fallback text vector", scanned.path)
            # Create a fallback chunk embedding based on file name and path for scanned/graphic PDFs
            fallback_text = scanned.file_name.replace("_", " ").replace("-", " ")
            vectors = self._text_embedder.encode_documents([fallback_text])
            self._vector_service.upsert_text_chunk(
                file_id=file_id,
                file_name=scanned.file_name,
                path=scanned.path,
                page_number=1,
                chunk_text=f"Document: {scanned.file_name}",
                chunk_index=0,
                vector=vectors[0],
            )

        self._metadata_service.upsert_file(
            file_name=scanned.file_name,
            file_type=FileType.document,
            extension=scanned.extension,
            path=scanned.path,
            size_bytes=scanned.size_bytes,
            chunk_count=len(chunks) if chunks else 1,
        )

    def _index_code(self, scanned: ScannedFile, file_id: str, root: str) -> None:
        parsed = parse_code(scanned.path, scanned.extension)
        chunks = chunk_code(parsed.text, parsed.language)

        if chunks:
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
