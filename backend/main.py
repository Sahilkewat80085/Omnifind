from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_files import router as files_router
from api.routes_index import router as index_router
from api.routes_search import router as search_router
from database.session import init_db
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.on_event("startup")
def on_startup() -> None:
    logger.info("%s starting up in '%s' mode", settings.app_name, settings.app_env)
    init_db()
    logger.info("Database initialized")
