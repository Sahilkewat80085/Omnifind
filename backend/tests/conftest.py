import pytest


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Points Qdrant local storage and SQLite at a per-test tmp directory.

    Both the settings object and the Qdrant client are process-wide
    lru_cache singletons (needed in production so indexing and search share
    one client and one storage lock). Tests must clear those caches so each
    test gets its own isolated storage instead of colliding on the real one.
    """
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant_local"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")

    from core.vectorstore.qdrant_client import _get_client
    from utils.config import get_settings

    get_settings.cache_clear()
    _get_client.cache_clear()

    # The environment variable alone does not move SQLite. `database.session`
    # builds its engine at import time, and every module that does
    # `from database.session import SessionLocal` holds a reference to that one
    # sessionmaker - so setting DATABASE_URL after the fact isolated nothing,
    # and a bare `SearchService()` in a test read the developer's real index.
    # That went unnoticed while search was vector-first and the real metadata
    # database happened to hold no passage text. Re-binding the shared
    # sessionmaker in place is what every holder of it actually sees.
    from sqlalchemy import create_engine

    from database.models import Base
    from database.session import SessionLocal, _resolve_sqlite_url

    test_engine = create_engine(
        _resolve_sqlite_url(f"sqlite:///{(tmp_path / 'test.db').as_posix()}"),
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=test_engine)
    previous_bind = SessionLocal.kw.get("bind")
    SessionLocal.configure(bind=test_engine)

    yield tmp_path

    SessionLocal.configure(bind=previous_bind)
    test_engine.dispose()
    try:
        _get_client().close()
    except Exception:
        pass
    _get_client.cache_clear()
    get_settings.cache_clear()
