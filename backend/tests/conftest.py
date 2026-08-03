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

    yield tmp_path

    try:
        _get_client().close()
    except Exception:
        pass
    _get_client.cache_clear()
    get_settings.cache_clear()
