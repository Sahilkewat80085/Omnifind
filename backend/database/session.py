from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base
from utils.config import BACKEND_ROOT, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _resolve_sqlite_url(url: str) -> str:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw_path = url[len(prefix):]
    if raw_path.startswith("/") or ":" in raw_path:
        return url  # already absolute
    return f"{prefix}{(BACKEND_ROOT / raw_path).resolve().as_posix()}"


engine = create_engine(
    _resolve_sqlite_url(settings.database_url),
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _add_missing_columns() -> None:
    """Add nullable columns that the model has and the existing table lacks.

    `create_all` only creates missing *tables* — it never alters one that
    already exists, so a new column would leave every query selecting a column
    SQLite does not have, and the app would fail on an index built before the
    change. This is not a general migration system: it only adds nullable
    columns, which is the one change SQLite can make cheaply and safely, and
    it is a deliberate alternative to pulling in Alembic for a desktop app
    whose whole database is one table.
    """
    inspector = inspect(engine)
    if not inspector.has_table("files"):
        return  # fresh install; create_all just built it correctly

    existing = {column["name"] for column in inspector.get_columns("files")}
    for column in Base.metadata.tables["files"].columns:
        if column.name in existing:
            continue
        if not column.nullable:
            # Nothing sensible to backfill a NOT NULL column with. Loud, and
            # rare enough to deal with by hand if it ever happens.
            logger.error("Cannot auto-add non-nullable column %r; a migration is needed", column.name)
            continue
        column_type = column.type.compile(dialect=engine.dialect)
        logger.info("Adding missing column files.%s (%s)", column.name, column_type)
        with engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE files ADD COLUMN {column.name} {column_type}"))


def init_db() -> None:
    (BACKEND_ROOT / "storage").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
