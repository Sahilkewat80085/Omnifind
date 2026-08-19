import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from core.parsers.code_parser import CODE_EXTENSIONS
from models.schemas.file_schemas import FileType
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | set(CODE_EXTENSIONS)

# Directories that are never the user's own work. Without this, pointing the
# scanner at any real project walks straight into node_modules or .venv and
# tries to index tens of thousands of third-party files — this backend's own
# .venv alone holds the whole of scikit-learn. Pruning them is what makes
# indexing a repository finish at all.
IGNORED_DIRECTORIES = {
    "node_modules",
    "site-packages",
    "vendor",
    "venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    "out",
    "target",
    "coverage",
    "bin",
    "obj",
    "migrations",
    "Pods",
    # OmniFind's own embedded vector store. Indexing the index is pure noise:
    # its meta.json is not source anyone searches for, and it grows every scan.
    "qdrant_local",
}


def is_ignored_directory(name: str) -> bool:
    # Every dot-directory goes too: .git, .venv, .next, .mypy_cache, .idea.
    # They are tooling state, not source anyone searches for.
    return name in IGNORED_DIRECTORIES or name.startswith(".")


def classify_extension(extension: str) -> FileType | None:
    ext = extension.lower()
    if ext in DOCUMENT_EXTENSIONS:
        return FileType.document
    if ext in IMAGE_EXTENSIONS:
        return FileType.image
    if ext in CODE_EXTENSIONS:
        return FileType.code
    return None


@dataclass(frozen=True)
class ScannedFile:
    path: str
    file_name: str
    extension: str
    file_type: FileType
    size_bytes: int


class FolderScanner:
    def scan(self, root: str) -> Iterator[ScannedFile]:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {root}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Not a folder: {root}")

        def _on_walk_error(error: OSError) -> None:
            logger.warning("Skipping unreadable directory: %s", error)

        max_code_bytes = get_settings().code_max_file_bytes

        for dirpath, dirnames, filenames in os.walk(root_path, onerror=_on_walk_error):
            # Assigning to the slice is what prunes the walk: os.walk reads
            # this list back to decide where to descend, so filtering a copy
            # would still visit every ignored directory.
            dirnames[:] = [d for d in dirnames if not is_ignored_directory(d)]

            for filename in filenames:
                extension = Path(filename).suffix.lower()
                file_type = classify_extension(extension)
                if file_type is None:
                    continue

                full_path = Path(dirpath) / filename
                try:
                    size_bytes = full_path.stat().st_size
                except OSError as exc:
                    logger.warning("Skipping unreadable file %s: %s", full_path, exc)
                    continue

                if file_type == FileType.code and size_bytes > max_code_bytes:
                    # Minified bundles and generated sources are real code but
                    # never the answer to a question, and one of them can be
                    # larger than an entire hand-written repository.
                    logger.info("Skipping oversized source file (%d bytes): %s", size_bytes, full_path)
                    continue

                yield ScannedFile(
                    path=str(full_path.resolve()),
                    file_name=filename,
                    extension=extension,
                    file_type=file_type,
                    size_bytes=size_bytes,
                )
