"""Top-k must count files, and the UI's type filter must beat the query's.

Two changes lock in here. Ranked chunks are collapsed to one row per file
*before* the top-k cut, because cutting chunks first let a single long file
consume the whole result list — the user asked for ten results and, once the
UI de-duplicates by file, saw one. And the Search page's type dropdown sends
an explicit `file_type`, which has to override the type word `detect_intent`
reads out of the wording rather than losing to it.
"""

from models.schemas.file_schemas import FileType
from models.schemas.search_schemas import CodeResult, DocumentResult, ImageResult
from services.search_service import SearchService, best_per_file


def _doc(file_id: str, similarity: float, chunk_index: int = 0) -> DocumentResult:
    return DocumentResult(
        file_id=file_id,
        file_name=f"{file_id}.pdf",
        path=f"/tmp/{file_id}.pdf",
        similarity=similarity,
        page_number=None,
        chunk_text="…",
        chunk_index=chunk_index,
    )


# ---------- one row per file, keeping the best passage ----------


def test_best_per_file_keeps_the_strongest_chunk_of_each_file():
    ranked = [_doc("a", 0.9, 3), _doc("a", 0.8, 7), _doc("b", 0.7, 0), _doc("a", 0.1, 9)]

    kept = best_per_file(ranked)

    assert [(r.file_id, r.chunk_index) for r in kept] == [("a", 3), ("b", 0)]


def test_best_per_file_preserves_rank_order():
    ranked = [_doc("a", 0.9), _doc("b", 0.6), _doc("c", 0.3)]

    assert [r.file_id for r in best_per_file(ranked)] == ["a", "b", "c"]


def test_one_long_file_cannot_consume_the_whole_result_list(isolated_env, tmp_path):
    """The regression: many matching chunks of one file, plus other files.

    Before the per-file collapse moved ahead of the cut, a top-k of 3 spent
    every slot on `long.txt` and the two other files never reached the screen.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import Base
    from services.indexing_service import IndexingService
    from services.metadata_service import MetadataService

    source = tmp_path / "source"
    source.mkdir()
    # Long enough to chunk many times over, every chunk about the same subject.
    (source / "long.txt").write_text(
        "\n\n".join(
            f"Section {n}. Database normalization removes redundant data and "
            f"splits tables so each fact about the schema is stored once."
            for n in range(40)
        ),
        encoding="utf-8",
    )
    (source / "second.txt").write_text(
        "Normalizing a relational database schema into third normal form.",
        encoding="utf-8",
    )
    (source / "third.txt").write_text(
        "Redundant data in tables and how normalization removes it.",
        encoding="utf-8",
    )

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    IndexingService(MetadataService(db)).index_folder(str(source))

    response = SearchService().search("database normalization", top_k=3)

    names = [r.file_name for r in response.results]
    assert len(names) == len(set(names)), f"same file listed twice: {names}"
    assert len(names) == 3, f"top_k=3 returned {len(names)} files: {names}"

    db.close()


# ---------- the explicit type filter ----------


def test_explicit_file_type_filters_when_the_query_names_nothing(isolated_env, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from PIL import Image

    from database.models import Base
    from services.indexing_service import IndexingService
    from services.metadata_service import MetadataService

    source = tmp_path / "source"
    source.mkdir()
    (source / "brochure.txt").write_text(
        "Guided tours across scenic mountain ranges and alpine lakes.",
        encoding="utf-8",
    )
    Image.new("RGB", (256, 256), (30, 90, 200)).save(source / "lake.png")

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    IndexingService(MetadataService(db)).index_folder(str(source))

    service = SearchService(db=db)


    # The filter is doing the work on image type.
    images = service.search("lake", file_type=FileType.image)
    assert images.filtered_to == FileType.image
    assert images.results, "the filter searched nothing"
    assert all(isinstance(r, ImageResult) for r in images.results)

    documents = service.search("scenic mountain ranges", file_type=FileType.document)
    assert documents.filtered_to == FileType.document
    assert all(isinstance(r, DocumentResult) for r in documents.results)


    db.close()


def test_explicit_file_type_overrides_the_word_in_the_query(isolated_env, tmp_path):
    """"code" in the wording says code; the dropdown says documents. It wins."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import Base
    from services.indexing_service import IndexingService
    from services.metadata_service import MetadataService

    source = tmp_path / "source"
    source.mkdir()
    (source / "handbook.txt").write_text(
        "The department code of conduct for laboratory safety and reporting.",
        encoding="utf-8",
    )
    (source / "safety.py").write_text(
        "def check_lab_safety(report):\n    return report.is_signed\n",
        encoding="utf-8",
    )

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    IndexingService(MetadataService(db)).index_folder(str(source))

    response = SearchService().search("code of conduct", file_type=FileType.document)

    assert response.filtered_to == FileType.document
    assert not any(isinstance(r, CodeResult) for r in response.results)
    assert response.results and response.results[0].file_name == "handbook.txt"

    db.close()
