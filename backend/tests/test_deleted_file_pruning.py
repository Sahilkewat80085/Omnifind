"""Deleting a file must remove it from the index, not just from the disk.

The bug these lock in: a folder scan only ever added what it found, so an
image deleted after indexing kept its SQLite row and its Qdrant vectors. It
went on ranking in search, and "Open" on the result failed with file not
found. Two layers fix it — a scan-time prune, and a disk check at query
time so the stale hit is gone before the next scan even runs.
"""

from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base
from services.indexing_service import IndexingService
from services.metadata_service import MetadataService
from services.search_service import SearchService


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_rescan_removes_a_deleted_file_from_metadata_and_vectors(isolated_env, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.txt").write_text("a note that stays put")
    (source / "gone.txt").write_text("a note about a person sitting on a train")

    db = _session(tmp_path)
    metadata_service = MetadataService(db)
    indexing_service = IndexingService(metadata_service)

    indexing_service.index_folder(str(source))
    assert metadata_service.get_stats().total_files == 2
    assert SearchService().search("person sitting on a train").results

    (source / "gone.txt").unlink()
    summary = indexing_service.index_folder(str(source))

    assert summary.removed == 1
    assert metadata_service.get_stats().total_files == 1
    assert metadata_service.get_by_path(str((source / "gone.txt").resolve())) is None

    # The vectors have to go too, or search keeps answering from them.
    hit_names = {r.file_name for r in SearchService().search("person sitting on a train").results}
    assert "gone.txt" not in hit_names

    db.close()


def test_deleted_image_disappears_from_search(isolated_env, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (320, 240), color=(120, 60, 30)).save(source / "train.png")

    db = _session(tmp_path)
    indexing_service = IndexingService(MetadataService(db))
    indexing_service.index_folder(str(source))

    (source / "train.png").unlink()

    # Before any re-scan: the vector is still in Qdrant, but the disk check
    # keeps it out of the results, so no user ever gets a dead "Open" button.
    assert all(Path(r.path).exists() for r in SearchService().search("a photo").results)

    indexing_service.index_folder(str(source))
    assert MetadataService(db).get_stats().total_files == 0

    db.close()


def test_prune_is_scoped_to_the_folder_being_scanned(isolated_env, tmp_path):
    """Indexing folder A must never evict what was indexed from folder B."""
    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    (folder_a / "one.txt").write_text("contents of the first folder")
    (folder_b / "two.txt").write_text("contents of the second folder")

    db = _session(tmp_path)
    metadata_service = MetadataService(db)
    indexing_service = IndexingService(metadata_service)

    indexing_service.index_folder(str(folder_a))
    indexing_service.index_folder(str(folder_b))
    assert metadata_service.get_stats().total_files == 2

    summary = indexing_service.index_folder(str(folder_a))

    assert summary.removed == 0
    assert metadata_service.get_stats().total_files == 2
    assert metadata_service.get_by_path(str((folder_b / "two.txt").resolve())) is not None

    db.close()


def test_a_file_skipped_by_the_scanner_but_still_on_disk_is_kept(isolated_env, tmp_path):
    """Only absence from the *disk* prunes — absence from a scan does not.

    A source file that grows past code_max_file_bytes stops being scanned,
    but it is still the user's file and must keep its entry.
    """
    source = tmp_path / "source"
    source.mkdir()
    big = source / "huge.py"
    big.write_text("def small():\n    return 1\n")

    db = _session(tmp_path)
    metadata_service = MetadataService(db)
    indexing_service = IndexingService(metadata_service)
    indexing_service.index_folder(str(source))
    assert metadata_service.get_stats().total_files == 1

    big.write_text("x = 1\n" * 400_000)  # now over the size limit, so skipped
    summary = indexing_service.index_folder(str(source))

    assert summary.indexed == 0
    assert summary.removed == 0
    assert metadata_service.get_by_path(str(big.resolve())) is not None

    db.close()
