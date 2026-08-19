"""A stated file type is a filter, not a ranking hint.

Searching "mountain image" and being handed a PDF that discusses mountains is
a wrong answer however strong the semantic match: the user named the type they
wanted. These lock in both halves — reading the type out of the query, and
letting it decide which partitions are searched at all.
"""

from core.query.intent import detect_intent
from models.schemas.file_schemas import FileType


def test_a_named_type_becomes_a_filter_and_leaves_the_query():
    intent = detect_intent("mountain image")

    assert intent.file_type == FileType.image
    # "image" describes the container, not the content — leaving it in only
    # blurs the vector the subject is matched with.
    assert intent.query == "mountain"


def test_stopwords_stranded_by_the_removal_are_trimmed():
    assert detect_intent("picture of a dog").query == "dog"
    assert detect_intent("image of mountains").query == "mountains"


def test_documents_and_code_are_recognised_too():
    assert detect_intent("invoice pdf").file_type == FileType.document
    assert detect_intent("the kalman filter code").file_type == FileType.code
    assert detect_intent("how does the login function work").file_type == FileType.code


def test_an_ambiguous_query_filters_nothing():
    """Two types named means neither is meant as a filter. Searching
    everything is never wrong, only unhelpful — silently hiding two thirds of
    the index would be."""
    intent = detect_intent("image processing code")

    assert intent.file_type is None
    assert intent.query == "image processing code"


def test_a_query_with_no_type_word_is_untouched():
    intent = detect_intent("how much did I pay in fees")

    assert intent.file_type is None
    assert intent.query == "how much did I pay in fees"


def test_content_words_are_not_mistaken_for_types():
    """The subject of a search must never be read as its container."""
    for query in ("travel destinations brochure", "class notes for databases",
                  "research method for the survey"):
        assert detect_intent(query).file_type is None, query


def test_a_bare_type_word_keeps_something_to_embed():
    intent = detect_intent("images")

    assert intent.file_type == FileType.image
    # Nothing would be left to search on, so the original wording stands and
    # the filter does the work.
    assert intent.query == "images"


def test_empty_and_punctuation_only_queries_are_safe():
    assert detect_intent("").file_type is None
    assert detect_intent("???").file_type is None


def test_search_returns_only_the_requested_type(isolated_env, tmp_path):
    """End to end: the reported bug. A brochure about mountains must not
    answer a search for a mountain *image*."""
    from PIL import Image
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import Base
    from models.schemas.search_schemas import DocumentResult
    from services.indexing_service import IndexingService
    from services.metadata_service import MetadataService
    from services.search_service import SearchService

    source = tmp_path / "mixed"
    source.mkdir()
    (source / "brochure.txt").write_text(
        "TRAVEL BROCHURE. Discover Emerald Valley: experience scenic mountains, "
        "rivers, guided hikes and cycling routes.",
        encoding="utf-8",
    )
    Image.new("RGB", (128, 128), (30, 120, 60)).save(source / "scene.png")

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    IndexingService(MetadataService(db)).index_folder(str(source))

    response = SearchService().search("mountain image")

    assert response.filtered_to == FileType.image
    assert not any(isinstance(r, DocumentResult) for r in response.results), (
        "a document answered a search that explicitly asked for an image"
    )
    # The query the user typed is echoed back unchanged, not the stripped one.
    assert response.query == "mountain image"
    db.close()


def test_an_unfiltered_search_still_reaches_every_type(isolated_env, tmp_path):
    """The filter must be opt-in by wording — a normal query is unaffected."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import Base
    from services.indexing_service import IndexingService
    from services.metadata_service import MetadataService
    from services.search_service import SearchService

    source = tmp_path / "docs"
    source.mkdir()
    (source / "fees.txt").write_text(
        "Total fee paid for the semester: 15,000 rupees.", encoding="utf-8"
    )

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    IndexingService(MetadataService(db)).index_folder(str(source))

    response = SearchService().search("how much was the fee")

    assert response.filtered_to is None
    assert response.results and response.results[0].file_name == "fees.txt"
    db.close()
