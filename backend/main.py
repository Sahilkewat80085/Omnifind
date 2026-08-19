from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_ask import router as ask_router
from api.routes_files import router as files_router
from api.routes_index import router as index_router
from api.routes_search import router as search_router
from database.session import SessionLocal, init_db
from services.folder_watcher_service import get_watcher_service
from services.metadata_service import MetadataService
from utils.config import get_settings
from utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(index_router)
app.include_router(search_router)
app.include_router(files_router)
app.include_router(ask_router)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "ai_enabled": bool(settings.gemini_api_key.strip()),
        "model": settings.gemini_model,
    }


@app.on_event("startup")
def on_startup() -> None:
    logger.info("%s starting up in '%s' mode", settings.app_name, settings.app_env)
    init_db()
    logger.info("Database initialized")

    # Resume real-time watching for all registered folders
    db = SessionLocal()
    try:
        watcher_service = get_watcher_service()
        watcher_service.start()
        watched_folders = MetadataService(db).list_watched_folders(only_active=True)
        for folder in watched_folders:
            watcher_service.watch_folder(folder.path)
            logger.info("Resumed background folder watch on startup: %s", folder.path)
    except Exception:
        logger.exception("Failed to resume watched folders on startup")
    finally:
        db.close()


@app.on_event("shutdown")
def on_shutdown() -> None:
    logger.info("Stopping background folder watchers")
    get_watcher_service().stop()
