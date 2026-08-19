from dataclasses import dataclass
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

# Extension → language label. This is also the definition of "what counts as a
# code file": the scanner derives its extension set from these keys, so adding
# a language here is the only edit needed to start indexing it.
LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".vue": "vue",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".json": "json",
}

CODE_EXTENSIONS = frozenset(LANGUAGE_BY_EXTENSION)

# Read enough to be confident about NUL bytes without pulling a large file
# into memory just to reject it.
_BINARY_SNIFF_BYTES = 8192


class BinaryFileError(ValueError):
    """Raised for a file with a source extension that holds binary content."""


@dataclass(frozen=True)
class CodeFile:
    language: str
    text: str


def detect_language(extension: str) -> str:
    return LANGUAGE_BY_EXTENSION.get(extension.lower(), "text")


def parse_code(path: str, extension: str) -> CodeFile:
    """Read a source file as text, preserving its exact line structure.

    Unlike the document parsers this returns one blob rather than pages, and
    deliberately does not strip or normalize: indentation *is* the syntax in
    Python and YAML, and line numbers have to survive so a search hit can point
    at a real location in the file.
    """
    file_path = Path(path)

    with file_path.open("rb") as handle:
        head = handle.read(_BINARY_SNIFF_BYTES)
    if b"\x00" in head:
        # A .h that is really a compiled header, a .json holding a binary
        # blob — decoding it would produce replacement-character soup that
        # embeds as meaningless noise.
        raise BinaryFileError(f"Binary content in source file: {path}")

    # utf-8 first (the overwhelming default for source), then cp1252 for files
    # written by older Windows editors, which is common in student projects.
    for encoding in ("utf-8", "cp1252"):
        try:
            text = file_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        logger.warning("Falling back to lossy decode for %s", path)
        text = file_path.read_text(encoding="utf-8", errors="replace")

    return CodeFile(language=detect_language(extension), text=text)
