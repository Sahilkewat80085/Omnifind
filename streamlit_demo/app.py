"""
OmniFind — Streamlit demo UI

Stand-in for the real Tauri + React frontend, which is blocked on installing
Node.js + Rust. This talks to the exact same FastAPI backend over HTTP — no
mock data, no shortcuts — so it's a legitimate way to demo the working engine
without a browser full of raw JSON.

Run:
    1) backend:  cd omnifind/backend  &&  uvicorn main:app --reload --port 8000
    2) this app: cd omnifind/streamlit_demo  &&  streamlit run app.py
"""

import time
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="OmniFind", page_icon="\U0001F50E", layout="wide")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:8000"

EXAMPLES_DIR = str(Path(__file__).resolve().parents[1] / "examples")

EXAMPLE_QUERIES = [
    "how much money was paid",
    "travel destinations brochure",
    "bill for goods received",
]


def api_url(path: str) -> str:
    return st.session_state.backend_url.rstrip("/") + path


def get_health() -> dict | None:
    try:
        r = requests.get(api_url("/health"), timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


# ---------- sidebar ----------
with st.sidebar:
    st.title("\U0001F50E OmniFind")
    st.caption("Semantic search over local files — demo UI")

    st.session_state.backend_url = st.text_input(
        "Backend URL", value=st.session_state.backend_url
    )

    health = get_health()
    if health:
        st.success(f"Backend online — {health.get('app')} ({health.get('env')})")
    else:
        st.error("Backend not reachable")
        st.code("cd omnifind\\backend\n.venv\\Scripts\\activate\nuvicorn main:app --reload --port 8000")

    st.divider()
    page = st.radio("Go to", ["Dashboard", "Index a folder", "Search"], label_visibility="collapsed")

    st.divider()
    st.caption(
        "This is a temporary UI. The real desktop app (Tauri + React) is "
        "scoped and will call this exact same API — no backend changes needed."
    )

if not health:
    st.warning("Start the FastAPI backend first (see sidebar), then reload this page.")
    st.stop()


# ---------- Dashboard ----------
if page == "Dashboard":
    st.header("Dashboard")

    try:
        stats = requests.get(api_url("/index/stats"), timeout=10).json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not load stats: {e}")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Files indexed", stats["total_files"])
    c2.metric("Documents", stats["total_documents"])
    c3.metric("Images", stats["total_images"])
    c4.metric("Chunks", stats["total_chunks"])
    c5.metric("Storage used", f'{stats["total_size_bytes"] / 1_000_000:.1f} MB')

    st.divider()
    st.subheader("What this proves")
    st.markdown(
        "- Two different data types (documents + images) live in **one** searchable index.\n"
        "- Every number above comes from the real SQLite database, live — nothing here is mocked.\n"
        "- Next step: pick a folder under **Index a folder** and search it under **Search**."
    )


# ---------- Index a folder ----------
elif page == "Index a folder":
    st.header("Index a folder")
    st.caption("Recursively scans a folder for PDF / DOCX / TXT / PNG / JPG and builds the searchable index.")

    folder = st.text_input("Folder path", value=EXAMPLES_DIR)
    start = st.button("Start indexing", type="primary")

    if start:
        try:
            r = requests.post(api_url("/index/scan"), json={"path": folder}, timeout=10)
            if r.status_code == 200:
                st.success("Indexing started.")
            elif r.status_code == 409:
                st.warning("An indexing job is already running — see progress below.")
            else:
                st.error(f"{r.status_code}: {r.json().get('detail', r.text)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

    st.divider()
    st.subheader("Progress")

    status_box = st.empty()
    watch = st.checkbox("Auto-refresh while indexing", value=True)

    try:
        status = requests.get(api_url("/index/status"), timeout=5).json()
    except requests.exceptions.RequestException as e:
        status = None
        st.error(f"Could not read status: {e}")

    if status:
        with status_box.container():
            if status["is_running"]:
                total = max(status["total"], 1)
                st.progress(min(status["processed"] / total, 1.0))
                st.write(f'{status["processed"]} / {status["total"]} files — currently: `{status["current_file"]}`')
            elif status["processed"] > 0:
                st.success(f'Done — indexed {status["processed"]} / {status["total"]} files.')
            else:
                st.info("No indexing job has run yet.")

            if status.get("last_error"):
                st.error(f'Last error: {status["last_error"]}')

        if status["is_running"] and watch:
            time.sleep(1.5)
            st.rerun()


# ---------- Search ----------
elif page == "Search":
    st.header("Search")
    st.caption('Ask in plain English — this matches *meaning*, not filenames or keywords.')

    st.write("Try:")
    cols = st.columns(len(EXAMPLE_QUERIES))
    example_clicked = None
    for col, q in zip(cols, EXAMPLE_QUERIES):
        if col.button(q):
            example_clicked = q

    with st.form("search_form"):
        query = st.text_input("Query", value=example_clicked or "", placeholder="e.g. how much did I pay in fees?")
        submitted = st.form_submit_button("Search", type="primary")

    if submitted or example_clicked:
        q = query or example_clicked
        try:
            r = requests.get(api_url("/search"), params={"q": q}, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Search failed: {e}")
            st.stop()

        results = data["results"]
        if not results:
            st.info("No results — index a folder first, or try a different query.")

        for res in results:
            with st.container(border=True):
                left, right = st.columns([4, 1])

                with left:
                    if res["result_type"] == "document":
                        page_note = f' — page {res["page_number"]}' if res.get("page_number") else ""
                        st.markdown(f'**\U0001F4C4 {res["file_name"]}**{page_note}')
                        st.caption(res["path"])
                        st.write(res["chunk_text"])
                    else:
                        st.markdown(f'**\U0001F5BC️ {res["file_name"]}**  ({res["width"]}×{res["height"]})')
                        st.caption(res["path"])
                        if Path(res["path"]).exists():
                            st.image(res["path"], width=240)

                with right:
                    st.metric("Match", f'{res["similarity"] * 100:.0f}%')
                    st.progress(min(max(res["similarity"], 0.0), 1.0))
                    if st.button("Open file", key=f'open-{res["file_id"]}-{res.get("chunk_index", 0)}'):
                        try:
                            requests.post(api_url("/files/open"), json={"path": res["path"]}, timeout=5)
                            st.toast(f'Opened {res["file_name"]}')
                        except requests.exceptions.RequestException as e:
                            st.error(f"Could not open file: {e}")
