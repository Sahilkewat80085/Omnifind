import threading
from dataclasses import dataclass

from database.session import SessionLocal
from services.indexing_service import IndexingService, IndexProgress
from services.metadata_service import MetadataService
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IndexJobState:
    is_running: bool = False
    processed: int = 0
    total: int = 0
    current_file: str | None = None
    last_error: str | None = None
    indexed_count: int = 0
    removed_count: int = 0


class IndexJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = IndexJobState()

    def get_state(self) -> IndexJobState:
        with self._lock:
            return IndexJobState(**self._state.__dict__)

    def start(self, path: str) -> bool:
        with self._lock:
            if self._state.is_running:
                return False
            self._state = IndexJobState(is_running=True)

        thread = threading.Thread(target=self._run, args=(path,), daemon=True)
        thread.start()
        return True

    def _run(self, path: str) -> None:
        db = SessionLocal()
        try:
            indexing_service = IndexingService(MetadataService(db))

            def on_progress(progress: IndexProgress) -> None:
                with self._lock:
                    self._state.processed = progress.processed
                    self._state.total = progress.total
                    self._state.current_file = progress.current_file

            summary = indexing_service.index_folder(path, on_progress=on_progress)

            with self._lock:
                self._state.indexed_count = summary.indexed
                self._state.removed_count = summary.removed
        except Exception as exc:
            logger.exception("Indexing job failed")
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            db.close()
            with self._lock:
                self._state.is_running = False


_job_manager = IndexJobManager()


def get_job_manager() -> IndexJobManager:
    return _job_manager
