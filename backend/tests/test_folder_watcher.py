import time
from pathlib import Path

from database.session import SessionLocal, init_db
from services.folder_watcher_service import FolderWatcherService
from services.metadata_service import MetadataService


def test_folder_watcher_integration(isolated_env):
    init_db()
    service = FolderWatcherService()
    service.start()

    try:
        watched_dir = Path(isolated_env) / "watched"
        watched_dir.mkdir(parents=True, exist_ok=True)

        assert service.watch_folder(str(watched_dir))
        assert str(watched_dir.resolve()) in service.list_watched()

        test_file = watched_dir / "sample_note.txt"
        test_file.write_text("Automated indexing real time content test.")

        # Queue upsert change and flush
        service._queue_change("upsert", str(test_file))
        time.sleep(1.0)
        service.process_pending_now()

        db1 = SessionLocal()
        try:
            meta1 = MetadataService(db1)
            record = meta1.get_by_path(str(test_file.resolve()))
            assert record is not None
            assert record.file_name == "sample_note.txt"
        finally:
            db1.close()

        # Delete file and queue deletion and flush
        if test_file.exists():
            test_file.unlink()
        service._queue_change("delete", str(test_file))
        time.sleep(1.0)
        service.process_pending_now()

        db2 = SessionLocal()
        try:
            meta2 = MetadataService(db2)
            record_after = meta2.get_by_path(str(test_file.resolve()))
            assert record_after is None
        finally:
            db2.close()
    finally:
        service.stop()
