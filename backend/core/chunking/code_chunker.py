import re
from dataclasses import dataclass

from utils.config import get_settings

settings = get_settings()

# Lines that begin a new named thing worth returning as its own search hit.
# These are intentionally shallow — a real parser per language would be a
# project of its own, and the cost of a missed boundary is only that two
# symbols share a chunk, not a wrong result.
_DEFINITION_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(async\s+def|def|class)\s+\w+"),
    "javascript": re.compile(
        r"^\s*(export\s+)?(default\s+)?(async\s+)?(function\s+\w+|class\s+\w+"
        r"|(const|let|var)\s+\w+\s*=\s*(async\s*)?(\([^)]*\)|\w+)\s*=>)"
    ),
    "java": re.compile(
        r"^\s*(public|private|protected|static|final|abstract|\s)*"
        r"(class|interface|enum|record|void|[\w<>\[\], ]+)\s+\w+\s*\("
    ),
    "go": re.compile(r"^\s*(func|type)\s+\w+"),
    "rust": re.compile(r"^\s*(pub\s+)?(async\s+)?(fn|struct|enum|impl|trait)\s+\w+"),
    "ruby": re.compile(r"^\s*(def|class|module)\s+\w+"),
    "php": re.compile(r"^\s*(public|private|protected|static|\s)*(function|class)\s+\w+"),
    "c": re.compile(r"^\w[\w\s\*]*\s+\**\w+\s*\([^;]*\)\s*\{?\s*$"),
    "csharp": re.compile(
        r"^\s*(public|private|protected|internal|static|async|override|\s)*"
        r"(class|interface|struct|enum|record|void|[\w<>\[\], ]+)\s+\w+\s*[\(\{]"
    ),
    "swift": re.compile(r"^\s*(public|private|internal|open|\s)*(func|class|struct|enum|protocol)\s+\w+"),
    "kotlin": re.compile(r"^\s*(public|private|internal|open|suspend|\s)*(fun|class|object|interface)\s+\w+"),
    "scala": re.compile(r"^\s*(private|protected|\s)*(def|class|object|trait)\s+\w+"),
    "sql": re.compile(r"^\s*(CREATE|ALTER)\s+(TABLE|VIEW|INDEX|PROCEDURE|FUNCTION)\s+", re.IGNORECASE),
    "css": re.compile(r"^[.#@\w][^{;]*\{\s*$"),
}
# Families that share a syntax close enough to reuse another language's rule.
_DEFINITION_PATTERNS["typescript"] = _DEFINITION_PATTERNS["javascript"]
_DEFINITION_PATTERNS["vue"] = _DEFINITION_PATTERNS["javascript"]
_DEFINITION_PATTERNS["cpp"] = _DEFINITION_PATTERNS["csharp"]

# Lines that belong to the definition *below* them, not the code above.
_ATTACHES_DOWNWARD = ("@", "#", "//", "///", "/*", "*", '"""', "'''", "<!--")


@dataclass(frozen=True)
class CodeChunk:
    chunk_index: int
    chunk_text: str
    line_start: int  # 1-based, inclusive — matches what an editor shows
    line_end: int
    symbol: str | None


def _find_boundaries(lines: list[str], language: str) -> list[int]:
    pattern = _DEFINITION_PATTERNS.get(language)
    if pattern is None:
        return [0]

    boundaries = [0]
    for i, line in enumerate(lines):
        if i == 0 or not pattern.match(line):
            continue
        # A decorator or doc comment sitting directly above a definition is
        # part of it. Without this the chunk for `def foo` would start below
        # its own @property, and the comment explaining it would be filed
        # under the previous function.
        start = i
        while start > 0:
            above = lines[start - 1].strip()
            if not above or not above.startswith(_ATTACHES_DOWNWARD):
                break
            start -= 1
        if start > boundaries[-1]:
            boundaries.append(start)
    return boundaries


def _extract_symbol(block: list[str], language: str) -> str | None:
    pattern = _DEFINITION_PATTERNS.get(language)
    if pattern is None:
        return None
    for line in block:
        if pattern.match(line):
            symbol = " ".join(line.split()).rstrip("{:").strip()
            return symbol[:100] if symbol else None
    return None


def chunk_code(
    source: str,
    language: str,
    max_lines: int | None = None,
    overlap_lines: int | None = None,
    min_lines: int | None = None,
) -> list[CodeChunk]:
    """Split source into chunks that follow the shape of the code.

    Boundaries land on function/class definitions where the language has a
    recognisable one, so a search hit is a whole symbol. Blocks longer than
    `max_lines` are then windowed with overlap, and blocks shorter than
    `min_lines` are merged into their neighbour.
    """
    size = max_lines or settings.code_chunk_max_lines
    overlap = overlap_lines or settings.code_chunk_overlap_lines
    minimum = min_lines if min_lines is not None else settings.code_chunk_min_lines
    if overlap >= size:
        raise ValueError("code_chunk_overlap_lines must be smaller than code_chunk_max_lines")

    lines = source.splitlines()
    if not any(line.strip() for line in lines):
        return []

    boundaries = _find_boundaries(lines, language)

    # (start, end) half-open line ranges, 0-based.
    blocks = [
        (start, boundaries[i + 1] if i + 1 < len(boundaries) else len(lines))
        for i, start in enumerate(boundaries)
    ]

    # Merge runs of short blocks — imports, constants, one-line helpers — so
    # they arrive as one meaningful chunk instead of a dozen thin vectors.
    #
    # Short blocks absorb *forward*, into what follows them. A bare `class X:`
    # line belongs with its first method, and a file's import block belongs
    # with the first thing that uses it; folding them backward would file the
    # class declaration under the function above it, where nobody searching
    # for that class would find it. Only a short block with nothing after it
    # falls back to merging into its predecessor.
    merged: list[tuple[int, int]] = []
    i = 0
    while i < len(blocks):
        start, end = blocks[i]
        while (end - start) < minimum and i + 1 < len(blocks) and (blocks[i + 1][1] - start) <= size:
            i += 1
            end = blocks[i][1]
        if (end - start) < minimum and merged and (end - merged[-1][0]) <= size:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
        i += 1

    chunks: list[CodeChunk] = []
    index = 0
    for start, end in merged:
        window_start = start
        while window_start < end:
            window_end = min(window_start + size, end)
            block = lines[window_start:window_end]

            if any(line.strip() for line in block):
                chunks.append(
                    CodeChunk(
                        chunk_index=index,
                        chunk_text="\n".join(block),
                        line_start=window_start + 1,
                        line_end=window_end,
                        symbol=_extract_symbol(block, language),
                    )
                )
                index += 1

            if window_end == end:
                break
            window_start = window_end - overlap

    return chunks


def build_embedding_text(chunk: CodeChunk, *, relative_path: str, language: str) -> str:
    """What actually gets embedded — not the bare source.

    The text model is bge-small-en, trained on English prose. Raw source
    embeds poorly against a question phrased in words, so each chunk is given
    a natural-language header: the file's path within the indexed folder and
    the symbol name, both of which are the terms people actually search for
    ("the rag service", "calibrate"). The stored chunk_text stays pure source
    so the UI shows real code.
    """
    header = f"{language} source file {relative_path}"
    if chunk.symbol:
        header += f", defining {chunk.symbol}"
    return f"{header}\n\n{chunk.chunk_text}"
