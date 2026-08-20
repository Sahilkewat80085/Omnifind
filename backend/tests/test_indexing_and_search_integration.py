import docx
import fitz
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base
from services.indexing_service import IndexingService
from services.metadata_service import MetadataService
from services.search_service import SearchService


def _make_fixture_folder(base):
    pdf_doc = fitz.open()
    pdf_doc.new_page().insert_text((72, 72), "Database normalization removes redundant data.")
    pdf_doc.save(str(base / "dbms_notes.pdf"))
    pdf_doc.close()

    d = docx.Document()
    d.add_paragraph("Meeting notes: reviewed the project timeline and next steps.")
    d.save(str(base / "meeting_notes.docx"))

    Image.new("RGB", (640, 480)).save(base / "whiteboard.png")


def test_index_folder_then_search_returns_expected_top_results(isolated_env, tmp_path):
    source_folder = tmp_path / "source"
    source_folder.mkdir()
    _make_fixture_folder(source_folder)

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    indexing_service = IndexingService(MetadataService(db))
    summary = indexing_service.index_folder(str(source_folder))
    assert summary.indexed == 3
    assert summary.removed == 0

    doc_response = SearchService(db=db).search("database normalization")
    assert doc_response.results[0].file_name == "dbms_notes.pdf"
    assert doc_response.results[0].result_type.value == "document"

    notes_response = SearchService(db=db).search("meeting notes")
    assert notes_response.results[0].file_name == "meeting_notes.docx"

    db.close()


def test_reindexing_same_file_does_not_duplicate_chunks(isolated_env, tmp_path):
    source_folder = tmp_path / "source"
    source_folder.mkdir()
    (source_folder / "notes.txt").write_text("first version of the notes")

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    metadata_service = MetadataService(db)
    indexing_service = IndexingService(metadata_service)

    indexing_service.index_folder(str(source_folder))
    indexing_service.index_folder(str(source_folder))  # re-scan same folder

    stats = metadata_service.get_stats()
    assert stats.total_files == 1
    assert stats.total_chunks == 1

    db.close()
