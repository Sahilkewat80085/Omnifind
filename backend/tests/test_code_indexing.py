"""Code as a third indexable type.

Code shares the text vector with documents rather than getting a model of its
own, so most of what needs locking in is the treatment *around* the embedding:
that source files are found without dragging in a dependency tree, that chunks
follow symbol boundaries instead of arbitrary windows, and that line structure
survives — none of which the document path does or should do.
"""

import pytest

from core.chunking.code_chunker import build_embedding_text, chunk_code
from core.parsers.code_parser import BinaryFileError, detect_language, parse_code
from core.scanner.folder_scanner import FolderScanner
from models.schemas.file_schemas import FileType

PYTHON_SOURCE = '''"""Module docstring."""
import os


@lru_cache(maxsize=1)
def cached_helper(value):
    """Explains itself."""
    return value * 2


class Calculator:
    def __init__(self, precision):
        self.precision = precision

    def add(self, a, b):
        total = a + b
        return round(total, self.precision)
'''


# ---------- scanning ----------


def test_source_files_are_classified_as_code(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')")
    (tmp_path / "ui.tsx").write_text("export const A = () => null;")
    (tmp_path / "notes.txt").write_text("prose")

    by_name = {r.file_name: r for r in FolderScanner().scan(str(tmp_path))}

    assert by_name["app.py"].file_type == FileType.code
    assert by_name["ui.tsx"].file_type == FileType.code
    # Plain text is still a document — adding code must not reclassify it.
    assert by_name["notes.txt"].file_type == FileType.document


def test_dependency_and_tooling_directories_are_never_walked(tmp_path):
    """The difference between indexing a repo in seconds and never finishing."""
    (tmp_path / "main.py").write_text("x = 1")
    for junk in ("node_modules", ".git", ".venv", "__pycache__", "dist", "site-packages"):
        (tmp_path / junk).mkdir()
        (tmp_path / junk / "vendored.py").write_text("y = 2")

    names = {r.file_name for r in FolderScanner().scan(str(tmp_path))}

    assert names == {"main.py"}


def test_nested_dependency_directories_are_pruned_too(tmp_path):
    """os.walk must not descend, not merely skip the files it finds there."""
    deep = tmp_path / "packages" / "web" / "node_modules" / "react" / "lib"
    deep.mkdir(parents=True)
    (deep / "index.js").write_text("module.exports = {};")
    (tmp_path / "packages" / "web" / "app.js").write_text("run();")

    names = {r.file_name for r in FolderScanner().scan(str(tmp_path))}

    assert names == {"app.js"}


def test_oversized_source_files_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_MAX_FILE_BYTES", "200")
    from utils.config import get_settings

    get_settings.cache_clear()
    try:
        (tmp_path / "small.py").write_text("x = 1")
        (tmp_path / "bundle.min.js").write_text("var a=1;" * 100)

        names = {r.file_name for r in FolderScanner().scan(str(tmp_path))}

        assert names == {"small.py"}
    finally:
        get_settings.cache_clear()


# ---------- parsing ----------


def test_parsing_preserves_indentation_and_line_breaks(tmp_path):
    """Prose chunking joins on whitespace; for code that erases the syntax."""
    path = tmp_path / "sample.py"
    path.write_text(PYTHON_SOURCE, encoding="utf-8")

    parsed = parse_code(str(path), ".py")

    assert parsed.language == "python"
    assert "    def add(self, a, b):" in parsed.text
    assert parsed.text.count("\n") == PYTHON_SOURCE.count("\n")


def test_binary_content_in_a_source_extension_is_rejected(tmp_path):
    path = tmp_path / "compiled.h"
    path.write_bytes(b"\x7fELF\x00\x00\x00\x00binary junk")

    with pytest.raises(BinaryFileError):
        parse_code(str(path), ".h")


def test_unknown_extension_falls_back_to_plain_text():
    assert detect_language(".py") == "python"
    assert detect_language(".TSX") == "typescript"
    assert detect_language(".zzz") == "text"


# ---------- chunking ----------


def test_chunks_follow_symbol_boundaries():
    chunks = chunk_code(PYTHON_SOURCE, "python")
    symbols = [c.symbol for c in chunks]

    assert "def cached_helper(value)" in symbols
    # A whole symbol, not a window through the middle of one: the function
    # body arrives with the signature that names it.
    helper = next(c for c in chunks if c.symbol == "def cached_helper(value)")
    assert "return value * 2" in helper.chunk_text
    # ...and nothing from the class below it leaked in.
    assert "class Calculator" not in helper.chunk_text


def test_a_small_class_is_kept_whole_rather_than_fragmented():
    """Its methods are each only two lines. Splitting them into separate
    vectors would bury the class in thin, near-identical chunks; the size
    floor is what keeps it as one findable unit."""
    chunks = chunk_code(PYTHON_SOURCE, "python")

    calculator = next(c for c in chunks if c.symbol == "class Calculator")
    assert "def __init__" in calculator.chunk_text
    assert "def add(self, a, b):" in calculator.chunk_text
    assert "return round(total, self.precision)" in calculator.chunk_text


def test_decorators_stay_attached_to_the_function_below_them():
    """Otherwise @lru_cache is filed under the previous symbol."""
    chunks = chunk_code(PYTHON_SOURCE, "python")

    owner = next(c for c in chunks if "def cached_helper" in c.chunk_text)
    assert "@lru_cache(maxsize=1)" in owner.chunk_text


def test_a_class_declaration_is_grouped_with_its_first_method():
    """Short blocks absorb forward: a bare `class X:` line must not be filed
    under the function above it, where nobody would find it."""
    chunks = chunk_code(PYTHON_SOURCE, "python")

    owner = next(c for c in chunks if "class Calculator:" in c.chunk_text)
    assert owner.symbol == "class Calculator"
    assert "def __init__" in owner.chunk_text


def test_line_numbers_point_at_real_locations():
    source_lines = PYTHON_SOURCE.splitlines()
    for chunk in chunk_code(PYTHON_SOURCE, "python"):
        assert chunk.line_start >= 1
        assert chunk.line_end <= len(source_lines)
        # The reported range must reproduce the chunk exactly, or "lines
        # 40-60" in the UI opens the wrong part of the file.
        expected = "\n".join(source_lines[chunk.line_start - 1 : chunk.line_end])
        assert chunk.chunk_text == expected


def test_long_blocks_are_windowed_with_overlap():
    source = "\n".join(f"line_{i} = {i}" for i in range(200))
    chunks = chunk_code(source, "python", max_lines=50, overlap_lines=10)

    assert len(chunks) > 1
    assert all(c.line_end - c.line_start < 50 for c in chunks)
    # Consecutive windows must overlap, so a symbol split across the seam is
    # still wholly present in one of them.
    assert chunks[1].line_start < chunks[0].line_end


def test_a_language_without_boundary_rules_still_chunks():
    chunks = chunk_code("alpha\nbeta\ngamma\n", "text")

    assert len(chunks) == 1
    assert chunks[0].symbol is None


def test_blank_and_empty_sources_produce_nothing():
    assert chunk_code("", "python") == []
    assert chunk_code("\n\n   \n", "python") == []


def test_embedding_text_adds_the_words_people_search_for():
    """bge-small is an English model; raw source gives it nothing to match a
    plain-language question against."""
    chunk = chunk_code(PYTHON_SOURCE, "python")[1]
    text = build_embedding_text(chunk, relative_path="core/calc.py", language="python")

    assert "python source file core/calc.py" in text
    assert chunk.symbol in text
    # The source itself still has to be in there — the header is a prefix,
    # not a replacement.
    assert chunk.chunk_text in text


# ---------- end to end ----------


def test_code_is_searchable_by_plain_language(isolated_env, tmp_path):
    """The thesis, applied to code: find it by what it does, not its name."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import Base
    from models.schemas.search_schemas import CodeResult
    from services.indexing_service import IndexingService
    from services.metadata_service import MetadataService
    from services.search_service import SearchService

    source = tmp_path / "repo"
    source.mkdir()
    (source / "auth.py").write_text(
        "def verify_password(raw, stored_hash):\n"
        "    return bcrypt.checkpw(raw.encode(), stored_hash)\n",
        encoding="utf-8",
    )
    (source / "shipping.py").write_text(
        "def estimate_delivery_window(distance_km, courier):\n"
        "    return distance_km / courier.average_speed\n",
        encoding="utf-8",
    )

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    IndexingService(MetadataService(db)).index_folder(str(source))

    query = "how do we check a user login credential?"
    results = SearchService(db=db).search(query).results

    top = results[0]
    assert isinstance(top, CodeResult)
    # No word of the query appears in the function it must find.
    assert top.file_name == "auth.py"
    assert top.language == "python"
    assert top.line_start == 1

    stats = MetadataService(db).get_stats()
    assert stats.total_code == 2
    db.close()


# ---------- relevance: keeping unrelated code out ----------
#
# Code sits far higher in bge's range than prose for any English query — each
# chunk carries an English header naming its path and symbol, and one file
# yields dozens of chunks. Scored through the document band this turned pure
# noise into a confident-looking 48-55% match against queries with nothing to
# do with the code. These lock in the separate band that fixed it.


def test_code_and_documents_are_calibrated_through_different_bands():
    from utils.config import get_settings

    settings = get_settings()

    # The floor sits in the measured gap between code's worst noise (0.593)
    # and its weakest true match (0.719).
    assert settings.search_code_score_floor > settings.search_text_score_floor
    assert 0.593 < settings.search_code_score_floor < 0.719


def test_noise_level_code_is_dropped_while_the_same_score_survives_as_prose():
    """The exact bug: 0.59 is noise for code but a real hit for a document."""
    from core.vectorstore.qdrant_client import SearchHit
    from services.search_service import _calibrate
    from utils.config import get_settings

    settings = get_settings()
    hit = SearchHit(score=0.59, payload={"file_name": "perception.py"})

    as_code = _calibrate(
        [hit],
        floor=settings.search_code_score_floor,
        ceil=settings.search_code_score_ceil,
        drop_below_floor=True,
    )
    as_document = _calibrate(
        [hit],
        floor=settings.search_text_score_floor,
        ceil=settings.search_text_score_ceil,
        drop_below_floor=False,
    )

    assert as_code == []
    assert as_document[0].score > 0.4


def test_a_genuine_code_match_still_scores_well():
    """The floor must not be so high that real matches vanish with the noise."""
    from core.vectorstore.qdrant_client import SearchHit
    from services.search_service import _calibrate
    from utils.config import get_settings

    settings = get_settings()
    # 0.719 was the weakest measured true match.
    hits = _calibrate(
        [SearchHit(score=0.719, payload={"file_name": "perception.py"})],
        floor=settings.search_code_score_floor,
        ceil=settings.search_code_score_ceil,
        drop_below_floor=True,
    )

    assert hits, "the weakest genuine match must survive the floor"
    # And clear the UI's own 0.3 presentation floor, or it is dropped anyway.
    assert hits[0].score > 0.3


def test_unrelated_query_returns_no_code_at_all(isolated_env, tmp_path):
    """End to end: a document query must not drag code in behind it."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import Base
    from models.schemas.search_schemas import CodeResult
    from services.indexing_service import IndexingService
    from services.metadata_service import MetadataService
    from services.search_service import SearchService

    source = tmp_path / "mixed"
    source.mkdir()
    (source / "kalman.py").write_text(
        "class LaneKalmanFilter:\n"
        "    def update(self, left_coeffs, right_coeffs, lane_width):\n"
        "        self.state = self.predict() + self.gain * left_coeffs\n"
        "        return self.state\n",
        encoding="utf-8",
    )
    (source / "brochure.txt").write_text(
        "Discover breathtaking mountain ranges, alpine lakes and hill station "
        "resorts on our guided Himalayan holiday tours.",
        encoding="utf-8",
    )

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    IndexingService(MetadataService(db)).index_folder(str(source))

    # Deliberately no type word: "photo"/"image" would filter to pictures and
    # test the intent filter instead of the code relevance floor, which is
    # what must do the work here.
    results = SearchService(db=db).search("scenic mountain ranges and alpine lakes").results

    assert not any(isinstance(r, CodeResult) for r in results), (
        "unrelated source code leaked into a query about scenery"
    )
    assert results and results[0].file_name == "brochure.txt"
    db.close()


def test_rag_applies_the_code_floor_not_the_prose_one():
    """Noise-level code must not reach the prompt either, where it burns
    context and gives the model irrelevant material to reason around."""
    from core.vectorstore.qdrant_client import SearchHit
    from services.rag_service import RagService
    from utils.config import get_settings

    settings = get_settings()
    service = RagService.__new__(RagService)
    service._settings = settings

    code_hit = SearchHit(score=0.59, payload={"file_type": "code"})
    doc_hit = SearchHit(score=0.59, payload={"file_type": "document"})

    assert service._floor_for(code_hit) == settings.search_code_score_floor
    assert service._floor_for(doc_hit) == settings.rag_min_similarity
    assert code_hit.score < service._floor_for(code_hit)
    assert doc_hit.score > service._floor_for(doc_hit)
