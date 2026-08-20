# This block runs before every other import on purpose — do not let an
# import sorter move it. huggingface_hub reads HF_HUB_OFFLINE into a constant
# at import time, so enforcing offline mode after sentence_transformers or
# open_clip has been imported does nothing at all. See utils/offline.py.
from utils.offline import enforce_offline_models  # isort:skip

enforce_offline_models()

import threading  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.requests import Request  # noqa: E402

from api.routes_ask import router as ask_router  # noqa: E402
from api.routes_files import router as files_router  # noqa: E402
from api.routes_index import router as index_router  # noqa: E402
from api.routes_search import router as search_router  # noqa: E402
from core.embeddings import readiness  # noqa: E402
from core.embeddings.errors import ModelsNotAvailableError  # noqa: E402
from database.session import SessionLocal, init_db  # noqa: E402
from services.folder_watcher_service import get_watcher_service  # noqa: E402
from services.metadata_service import MetadataService  # noqa: E402
from utils.config import BACKEND_ROOT, get_settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

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


@app.exception_handler(ModelsNotAvailableError)
def _models_missing(request: Request, exc: ModelsNotAvailableError) -> JSONResponse:
    """503, not 500: setup is incomplete, the server is not broken.

    The message names the fetch script, and the frontend already surfaces
    `detail` verbatim, so the user is told what to run rather than being shown
    a stack trace about an unreachable host.
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, object]:
    status = readiness.get_status()
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "ai_enabled": bool(settings.gemini_api_key.strip()),
        "model": settings.gemini_model,
        # Whether search can actually run. "pending" during the few seconds of
        # start-up warm-up, "unavailable" only when the one-time model download
        # never completed. Everything else here can still say "ok".
        "models_state": status.state,
        "models_detail": status.detail,
    }


@app.on_event("startup")
def on_startup() -> None:
    logger.info("%s starting up in '%s' mode", settings.app_name, settings.app_env)
    init_db()
    logger.info("Database initialized")

    # In a thread so a cold model load does not hold the port shut for several
    # seconds, and daemon so it can never keep a shutdown hanging. It also runs
    # before the watcher indexes anything, which means the model caches are
    # populated once here instead of racing to load twice.
    threading.Thread(target=readiness.warm_up, name="model-warmup", daemon=True).start()

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


# ---------------------------------------------------------------------------
# The built frontend, served by the backend itself.
#
# This is what makes the desktop window possible: one process, one origin. The
# UI stops being "a website on localhost the user has to visit" and the API
# stops needing a hard-coded port, because the page can just call its own
# origin. It also settles the file:// problem the frontend was designed around
# — the bundle is served over http like any other page, so relative asset paths
# and <img> tags pointing at /files/{id}/raw both behave normally.
#
# Mounted LAST on purpose. A mount at "/" catches every path that no earlier
# route matched, so registering it above the routers would swallow /search and
# /health and serve them index.html.
FRONTEND_DIST = BACKEND_ROOT.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    # html=True serves index.html for "/" — enough here because the UI has no
    # router and therefore no deep links to fall back for.
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")
    logger.info("Serving bundled frontend from %s", FRONTEND_DIST)
else:
    # Normal during development: `npm run dev` serves the UI on :5173 and talks
    # to this process over CORS. Only the desktop build needs the bundle.
    logger.info(
        "No frontend build at %s - API only. Run 'npm run build' in frontend/ "
        "to serve the UI from here.",
        FRONTEND_DIST,
    )
