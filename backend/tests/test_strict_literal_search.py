"""Search answers a containment question, and answers it in two tiers.

The behaviour these lock in, in the user's own words: searching a college name
returns the files with that college name in them and nothing else, and a file
*named* after what was searched sits above a file that merely mentions it.

Before this, ranking fused a lexical list and a semantic list by reciprocal
rank, so a file that never contained the words could out-rank one that did -
and a filename match had no guaranteed advantage over a passage match. Both are
wrong answers to "show me the files with my name in them".
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.query.literal import StemIndex, extract_literal_terms, stem
from database.models import Base
from models.schemas.file_schemas import FileType
from services.indexing_service import IndexingService
from services.metadata_service import MetadataService
from services.search_service import SearchService


COLLEGE = "Sinhgad College of Engineering"


@pytest.fixture
def corpus(isolated_env, tmp_path):
    """Three files that mention the college, and two that do not."""
    source = tmp_path / "source"
    source.mkdir()

    (source / "Sinhgad_College_Certificate.txt").write_text(
        "This certifies attendance for the academic year.", encoding="utf-8"
    )
    (source / "transcript.txt").write_text(
        f"Issued by {COLLEGE}, Pune. Semester six result sheet.", encoding="utf-8"
    )
    (source / "colleges_shortlist.txt").write_text(
        "Colleges considered: Sinhgad, COEP, VIT. Engineering seats compared.",
        encoding="utf-8",
    )
    # Same subject, none of the words - the file the old ranking would return.
    (source / "admission_essay.txt").write_text(
        "My engineering education shaped how I approach technical problems.",
        encoding="utf-8",
    )
    (source / "recipe.txt").write_text(
        "Boil the rice for twelve minutes and drain.", encoding="utf-8"
    )

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    IndexingService(MetadataService(db)).index_folder(str(source))
    yield db
    db.close()


# ---------- the gate ----------


def test_only_files_containing_the_words_come_back(corpus):
    names = {r.file_name for r in SearchService(db=corpus).search("Sinhgad College").results}

    assert names == {
        "Sinhgad_College_Certificate.txt",
        "transcript.txt",
        "colleges_shortlist.txt",
    }
    assert "admission_essay.txt" not in names, (
        "a file about engineering, containing none of the words, was returned"
    )
    assert "recipe.txt" not in names


def test_every_word_is_required_so_more_words_narrow_the_search(corpus):
    """The deliberate edge of the rule, and the one worth knowing about.

    Only the transcript carries all four words of the full college name, so
    only the transcript comes back - a certificate *named* for the college is
    excluded for lacking the word "Engineering". That is the honest reading of
    "don't show me files that don't have my college name", and it is what stops
    another college's brochure answering this query. Searching fewer words
    widens it again, which is the remedy and the second half of this test.
    """
    exact = {r.file_name for r in SearchService(db=corpus).search(COLLEGE).results}
    assert "Sinhgad_College_Certificate.txt" not in exact
    # The shortlist keeps its place: the words may be scattered through a file,
    # they just all have to be in it. Chunk boundaries are not the user's problem.
    assert exact == {"transcript.txt", "colleges_shortlist.txt"}

    fewer = {r.file_name for r in SearchService(db=corpus).search("Sinhgad").results}
    assert "Sinhgad_College_Certificate.txt" in fewer
    assert len(fewer) > len(exact)


def test_a_name_the_index_has_never_seen_returns_nothing(corpus):
    response = SearchService(db=corpus).search("Bhawarthi Institute of Technology")

    assert response.results == []


def test_semantic_similarity_cannot_admit_a_file_on_its_own(corpus):
    """The whole point: closeness in meaning reorders, it never adds."""
    response = SearchService(db=corpus).search("engineering education background")

    for result in response.results:
        haystack = (result.file_name + " " + getattr(result, "chunk_text", "")).lower()
        assert "engineering" in haystack or "education" in haystack


# ---------- the two tiers ----------


def test_a_file_named_for_the_query_outranks_a_file_that_mentions_it(corpus):
    results = SearchService(db=corpus).search("Sinhgad").results

    assert results, "no file matched a name that three files carry"
    assert results[0].file_name == "Sinhgad_College_Certificate.txt", (
        f"name match did not lead: {[r.file_name for r in results]}"
    )
    # The other two carry it in their text, so they follow rather than vanish.
    assert {r.file_name for r in results[1:]} == {"transcript.txt", "colleges_shortlist.txt"}


def test_a_name_match_is_never_shown_as_a_weak_hit(corpus):
    top = SearchService(db=corpus).search("Sinhgad").results[0]

    assert top.similarity >= 0.85


# ---------- words the index has never seen ----------


def test_an_unfindable_word_is_dropped_rather_than_emptying_the_results(corpus):
    """One stray word must not delete an otherwise good match.

    "how much was the college fee" is a fair question even when no file says
    "fee" - dropping the unfindable word is what BM25 does with a zero document
    frequency term. Requiring it would make a single typo return nothing.
    """
    response = SearchService(db=corpus).search("Sinhgad College accreditation")

    assert response.results
    assert "accreditation" in response.ignored_terms
    assert "sinhgad" not in response.ignored_terms


def test_a_quoted_phrase_is_all_or_nothing(corpus):
    exact = SearchService(db=corpus).search(f'"{COLLEGE}"')

    assert [r.file_name for r in exact.results] == ["transcript.txt"], (
        "quoting must demand the phrase whole, not its words scattered"
    )


# ---------- word forms ----------


def test_plurals_and_verb_forms_count_as_the_same_word(corpus):
    """A gate that rejects "colleges" for "college" is not strict, just broken."""
    names = {r.file_name for r in SearchService(db=corpus).search("colleges").results}

    assert "colleges_shortlist.txt" in names
    assert "Sinhgad_College_Certificate.txt" in names


@pytest.mark.parametrize(
    "one,other",
    [
        ("college", "colleges"),
        ("note", "notes"),
        ("fee", "fees"),
        ("normalization", "normalizing"),
        ("table", "tables"),
    ],
)
def test_inflections_of_one_word_share_a_stem(one, other):
    assert stem(one) == stem(other)


def test_a_longer_word_starting_with_the_query_matches_but_not_the_reverse():
    index = StemIndex()
    index.add("Redundant data in tables and how normalization removes it.")

    # "check" finding "checkpw" is the behaviour that makes code searchable.
    code = StemIndex()
    code.add("return bcrypt.checkpw(raw.encode(), stored_hash)")
    assert code.contains(stem("check"))

    # But "database" must not be satisfied by a file that only says "data".
    assert index.contains(stem("data"))
    assert not index.contains(stem("database"))


def test_short_words_demand_an_exact_match():
    index = StemIndex()
    index.add("The airport terminal opened in March.")

    assert not index.contains(stem("ai")), "a two-letter query prefix-matched a word"


# ---------- queries that demand nothing literal ----------


def test_a_bare_type_word_still_lists_that_type(corpus):
    """"documents" names a container and no subject - there is nothing to demand."""
    terms = extract_literal_terms("documents")

    assert terms.is_empty

    response = SearchService(db=corpus).search("documents")
    assert response.results, "a type-only query returned nothing to browse"


def test_stopwords_are_not_required_of_a_file(corpus):
    terms = extract_literal_terms("how much was the fee")

    assert terms.tokens == ("fee",)


# ---------- the escape hatch ----------


def test_turning_strict_matching_off_restores_meaning_only_ranking(corpus, monkeypatch):
    from utils.config import get_settings

    monkeypatch.setenv("SEARCH_STRICT_LEXICAL", "false")
    get_settings.cache_clear()
    try:
        service = SearchService(db=corpus)
        assert service._settings.search_strict_lexical is False
        # Nothing is required, so a file matching only by meaning may return.
        response = service.search("Bhawarthi Institute of Technology")
        assert isinstance(response.results, list)
    finally:
        get_settings.cache_clear()


def test_the_type_filter_still_narrows_a_literal_search(corpus):
    response = SearchService(db=corpus).search("Sinhgad", file_type=FileType.document)

    assert response.filtered_to == FileType.document
    assert response.results
    assert all(r.result_type == FileType.document for r in response.results)
