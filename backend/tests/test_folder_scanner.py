import pytest

from core.scanner.folder_scanner import FolderScanner
from models.schemas.file_schemas import FileType


def test_scan_recurses_and_filters_by_extension(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "doc.pdf").write_text("fake pdf")
    (tmp_path / "data.csv").write_text("not supported")
    (tmp_path / "sub" / "pic.PNG").write_text("fake png")

    results = list(FolderScanner().scan(str(tmp_path)))
    names = {r.file_name for r in results}

    assert names == {"notes.txt", "doc.pdf", "pic.PNG"}

    by_name = {r.file_name: r for r in results}
    assert by_name["notes.txt"].file_type == FileType.document
    assert by_name["pic.PNG"].file_type == FileType.image
    assert by_name["pic.PNG"].extension == ".png"


def test_scan_raises_for_missing_folder(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        list(FolderScanner().scan(str(missing)))
