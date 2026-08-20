"""OmniFind as a desktop application.

    python desktop.py

Starts the backend inside this process and opens it in a native window. The
user never sees a URL, a terminal, or a browser.

Why one process rather than a shell supervising a server: the backend is
already Python, so a separate window process would only add a lifecycle to get
wrong - starting the server, polling for readiness, noticing when it dies,
killing it on close, and leaving an orphan holding the Qdrant lock when any of
that fails. Running uvicorn on a thread in the same interpreter makes the
server's lifetime the window's lifetime by construction.
"""

# Offline enforcement has to happen before any model library is imported, and
# importing main is what pulls those in. See utils/offline.py.
from utils.offline import enforce_offline_models  # isort:skip

enforce_offline_models()

import socket  # noqa: E402
import threading  # noqa: E402

import uvicorn  # noqa: E402
import webview  # noqa: E402

from main import app  # noqa: E402
from utils.config import get_settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 850
# Below this the sidebar and a result card cannot both fit, and the layout
# starts overlapping rather than reflowing.
MIN_WIDTH = 940
MIN_HEIGHT = 600


def _bind_free_port() -> socket.socket:
    """Reserve a port by binding it, and hand uvicorn the socket itself.

    Asking the OS for a free port and then passing the *number* to uvicorn
    leaves a window in which something else can take it. Passing the bound
    socket closes that window: the port cannot be reassigned because this
    process never lets go of it.

    An OS-assigned port also means the desktop app does not fight a backend
    the user already has running on 8000 during development.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    return sock


def main() -> int:
    settings = get_settings()

    sock = _bind_free_port()
    port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    logger.info("Desktop backend bound to %s", url)

    server = uvicorn.Server(
        uvicorn.Config(app, log_level=settings.log_level.lower(), access_log=False)
    )

    # Daemon so a wedged server thread can never keep the app alive after the
    # window is gone - the user closed it, and a process with no window and no
    # way to stop it is the worst failure mode a desktop app has.
    thread = threading.Thread(
        target=lambda: server.run(sockets=[sock]), name="uvicorn", daemon=True
    )
    thread.start()

    # No readiness poll before opening the window. The page is served by this
    # same server, so if it is not up yet the webview simply retries; showing
    # the window immediately makes the app feel like it started instantly,
    # while models warm up in the background. /health reports models_state and
    # the UI banners it, so "starting" is visible without blocking on it.
    webview.create_window(
        settings.app_name,
        url,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
        text_select=True,
    )

    # Blocks on the main thread until the window is closed - a GUI event loop
    # has to own the main thread on Windows, which is why the server is the
    # part that got moved to a thread and not the other way round.
    webview.start()

    logger.info("Window closed, shutting down")
    server.should_exit = True
    thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
