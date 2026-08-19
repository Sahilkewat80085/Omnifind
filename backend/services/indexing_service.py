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


ProgressCallback = Callable[[IndexProgress], None]


class IndexingService:
    def __init__(self, metadata_service: MetadataService) -> None:
        self._metadata_service = metadata_service
        self._vector_service = VectorService()
        self._text_embedder = TextEmbeddingService()
        self._image_embedder = ImageEmbeddingService()
        self._vector_service.ensure_collection()

    def index_folder(self, root: str, on_progress: ProgressCallback | None = None) -> int:
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

        return indexed_count

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
