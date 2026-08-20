from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base, ChunkRecord, FileRecord
from models.schemas.file_schemas import FileType
from services.metadata_service import MetadataService, normalize_content_for_dedup
from services.search_service import SearchService, is_exact_term_query, collapse_near_duplicates
from models.schemas.search_schemas import DocumentResult


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_exact_term_query_classification():
    assert is_exact_term_query("AWS") is True
    assert is_exact_term_query("SQL") is True
    assert is_exact_term_query("DBMS") is True
    assert is_exact_term_query('"DealBridge"') is True
    assert is_exact_term_query("v1.5") is True
    assert is_exact_term_query("aws") is True

    # Natural language queries should NOT be classified as exact terms
    assert is_exact_term_query("cloud infrastructure costs") is False
    assert is_exact_term_query("how to setup database") is False
    assert is_exact_term_query("sustainable happiness") is False


def test_aws_query_returns_zero_when_not_in_corpus(in_memory_db, tmp_path):
    """Acceptance Criterion 1: AWS query returns 0 results when no document has AWS."""
    search_service = SearchService(db=in_memory_db)

    # Index unrelated document in SQLite
    doc_path = tmp_path / "synopsis.pdf"
    doc_path.write_text("This is an unrelated academic synopsis.", encoding="utf-8")
    meta = MetadataService(in_memory_db)
    meta.upsert_file(
        file_name="synopsis.pdf",
        file_type=FileType.document,
        extension=".pdf",
        path=str(doc_path),
        size_bytes=100,
    )
    meta.upsert_chunks(
        file_id="syn-1",
        file_name="synopsis.pdf",
        file_type=FileType.document,
        path=str(doc_path),
        chunks=[{"chunk_index": 0, "chunk_text": "Expected outcome of intelligent retrieval"}],
    )

    response = search_service.search("AWS", file_type=FileType.document)
    assert len(response.results) == 0, "Expected zero results for non-matching exact term query 'AWS'"


def test_aws_query_returns_matching_document(in_memory_db, tmp_path):
    """Acceptance Criterion 2: AWS query returns document when AWS is present."""
    search_service = SearchService(db=in_memory_db)

    aws_doc_path = tmp_path / "cloud_guide.pdf"
    aws_doc_path.write_text("Deploying architecture on AWS cloud services.", encoding="utf-8")
    meta = MetadataService(in_memory_db)
    file_rec = meta.upsert_file(
        file_name="cloud_guide.pdf",
        file_type=FileType.document,
        extension=".pdf",
        path=str(aws_doc_path),
        size_bytes=200,
    )
    meta.upsert_chunks(
        file_id=file_rec.id,
        file_name="cloud_guide.pdf",
        file_type=FileType.document,
        path=str(aws_doc_path),
        chunks=[{"chunk_index": 0, "chunk_text": "Deploying architecture on AWS cloud services."}],
    )

    response = search_service.search("AWS", file_type=FileType.document)
    assert len(response.results) >= 1
    assert response.results[0].file_name == "cloud_guide.pdf"
    assert response.results[0].match_source in ("lexical", "hybrid")


def test_near_duplicate_file_collapsing():
    """Acceptance Criterion 4: Near-duplicate synopsis files are collapsed into one card."""
    chunk_content = "OmniFind is an intelligent context-aware retrieval system."

    res1 = DocumentResult(
        file_id="f1",
        file_name="Final_synopsis_HOD_ma'am.pdf",
        path="/docs/Final_synopsis_HOD_ma'am.pdf",
        similarity=0.92,
        page_number=1,
        chunk_text=chunk_content,
        chunk_index=0,
    )
    res2 = DocumentResult(
        file_id="f2",
        file_name="omnifind synopsis final.pdf",
        path="/docs/omnifind synopsis final.pdf",
        similarity=0.91,
        page_number=1,
        chunk_text=chunk_content,
        chunk_index=0,
    )
    res3 = DocumentResult(
        file_id="f3",
        file_name="omnifind.pdf",
        path="/docs/omnifind.pdf",
        similarity=0.90,
        page_number=1,
        chunk_text=chunk_content,
        chunk_index=0,
    )

    collapsed = collapse_near_duplicates([res1, res2, res3])
    assert len(collapsed) == 1
    assert collapsed[0].file_name == "Final_synopsis_HOD_ma'am.pdf"
    assert collapsed[0].duplicate_count == 2
    assert "omnifind synopsis final.pdf" in collapsed[0].duplicate_files
    assert "omnifind.pdf" in collapsed[0].duplicate_files
