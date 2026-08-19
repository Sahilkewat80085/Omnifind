import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from core.scanner.folder_scanner import classify_extension, is_ignored_directory
from database.session import SessionLocal
from services.indexing_service import IndexingService
from services.metadata_service import MetadataService
from utils.logger import get_logger

logger = get_logger(__name__)

DEBOUNCE_SECONDS = 0.5


class _FolderEventHandler(FileSystemEventHandler):
    def __init__(self, root_folder: str, queue_change: Callable[[str, str], None]) -> None:
        super().__init__()
        self.root_folder = root_folder
        self.queue_change = queue_change

    def _should_ignore(self, path: str) -> bool:
        p = Path(path)
        for part in p.parts:
            if is_ignored_directory(part):
                return True
        return False

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory or self._should_ignore(event.src_path):
            return
        if classify_extension(Path(event.src_path).suffix):
            self.queue_change("upsert", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or self._should_ignore(event.src_path):
            return
        if classify_extension(Path(event.src_path).suffix):
            self.queue_change("upsert", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or self._should_ignore(event.src_path):
            return
        self.queue_change("delete", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if not self._should_ignore(event.src_path):
            self.queue_change("delete", event.src_path)
        dest_path = getattr(event, "dest_path", None)
        if dest_path and not self._should_ignore(dest_path):
            if classify_extension(Path(dest_path).suffix):
                self.queue_change("upsert", dest_path)


class FolderWatcherService:
    def __init__(self) -> None:
        self._observer: Observer | None = None
        self._watched_roots: dict[str, object] = {}
        self._pending_events: dict[str, tuple[str, float]] = {}  # path -> (action, timestamp)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._observer is not None:
                return
            self._observer = Observer()
            self._observer.start()
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._debounce_worker, daemon=True)
            self._worker_thread.start()
            logger.info("FolderWatcherService background engine started")

    def stop(self) -> None:
        with self._lock:
            if self._observer is None:
                return
            self._observer.stop()
            self._observer.join(timeout=3)
            self._observer = None
            self._watched_roots.clear()
            self._stop_event.set()
            logger.info("FolderWatcherService stopped")

    def watch_folder(self, folder_path: str) -> bool:
        resolved = str(Path(folder_path).resolve())
        if not os.path.isdir(resolved):
            return False

        self.start()

        with self._lock:
            if resolved in self._watched_roots:
                return True
            handler = _FolderEventHandler(resolved, self._queue_change)
            watch = self._observer.schedule(handler, resolved, recursive=True)
            self._watched_roots[resolved] = watch
            logger.info("Watching folder for automatic changes: %s", resolved)
            return True

    def unwatch_folder(self, folder_path: str) -> bool:
        resolved = str(Path(folder_path).resolve())
        with self._lock:
            if resolved not in self._watched_roots or self._observer is None:
                return False
            watch = self._watched_roots.pop(resolved)
            try:
                self._observer.unschedule(watch)
                logger.info("Unscheduled watch for folder: %s", resolved)
                return True
            except Exception:
                logger.exception("Failed to unschedule watch for: %s", resolved)
                return False

    def list_watched(self) -> list[str]:
        with self._lock:
            return list(self._watched_roots.keys())

    def _queue_change(self, action: str, file_path: str) -> None:
        resolved = str(Path(file_path).resolve())
        with self._lock:
            self._pending_events[resolved] = (action, time.time())

    def _find_root_for_file(self, file_path: str) -> str | None:
        p = Path(file_path)
        with self._lock:
            for root in self._watched_roots:
                if p.is_relative_to(Path(root)):
                    return root
        return None

    def process_pending_now(self) -> None:
        """Flushes and processes all currently queued file changes immediately."""
        with self._lock:
            ready_tasks = list(self._pending_events.items())
            self._pending_events.clear()

        if not ready_tasks:
            return

        db = SessionLocal()
        try:
            indexing_service = IndexingService(MetadataService(db))
            for path, (action, _) in ready_tasks:
                if action == "delete":
                    indexing_service.remove_single_file(path)
                elif action == "upsert":
                    root = self._find_root_for_file(path)
                    indexing_service.index_single_file(path, root_folder=root)
        except Exception:
            logger.exception("Error processing real-time file watcher queue")
        finally:
            db.close()

    def _debounce_worker(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.2)
            now = time.time()
            ready_tasks = []

            with self._lock:
                for path, (action, timestamp) in list(self._pending_events.items()):
                    if now - timestamp >= DEBOUNCE_SECONDS:
                        ready_tasks.append((path, action))
                        del self._pending_events[path]

            if not ready_tasks:
                continue

            db = SessionLocal()
            try:
                indexing_service = IndexingService(MetadataService(db))
                for path, action in ready_tasks:
                    if action == "delete":
                        indexing_service.remove_single_file(path)
                    elif action == "upsert":
                        root = self._find_root_for_file(path)
                        indexing_service.index_single_file(path, root_folder=root)
            except Exception:
                logger.exception("Error processing real-time file watcher queue")
            finally:
                db.close()


_watcher_service = FolderWatcherService()


def get_watcher_service() -> FolderWatcherService:
    return _watcher_service
