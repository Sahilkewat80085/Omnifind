"""The desktop shell: one process, one origin, no browser.

What can silently break here is the mount that serves the built UI. It sits at
"/", which catches every path no earlier route matched - so if it is ever
registered above the routers, /search and /health start returning index.html
with a 200 and the app looks like it is running while every request quietly
returns a web page instead of JSON.
"""

import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BACKEND_ROOT.parent / "frontend" / "dist"


@pytest.fixture(scope="module")
def client():
    from main import app

    return TestClient(app)


def test_api_routes_are_not_shadowed_by_the_ui_mount(client):
    """The regression that would make the whole app look mysteriously broken."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["status"] == "ok"


@pytest.mark.skipif(not FRONTEND_DIST.is_dir(), reason="frontend not built")
def test_the_built_ui_is_served_at_the_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<div id=\"root\">" in response.text


@pytest.mark.skipif(not FRONTEND_DIST.is_dir(), reason="frontend not built")
def test_the_production_bundle_does_not_hard_code_a_backend_port():
    """The desktop launcher binds an OS-assigned port, so a bundle that still
    pointed at 8000 would load its own UI and then fail every API call."""
    bundles = list((FRONTEND_DIST / "assets").glob("*.js"))
    assert bundles, "no built JS found"

    for bundle in bundles:
        assert "127.0.0.1:8000" not in bundle.read_text(encoding="utf-8", errors="ignore")


def test_free_port_is_handed_over_as_a_bound_socket():
    """Reserving a port and passing the *number* to uvicorn leaves a window in
    which another process can take it. Passing the socket closes that window."""
    pytest.importorskip("webview")
    from desktop import _bind_free_port

    sock = _bind_free_port()
    try:
        host, port = sock.getsockname()

        assert host == "127.0.0.1"
        assert port > 0

        # Still held: binding it again must fail, which is the guarantee.
        rival = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                rival.bind(("127.0.0.1", port))
        finally:
            rival.close()
    finally:
        sock.close()
