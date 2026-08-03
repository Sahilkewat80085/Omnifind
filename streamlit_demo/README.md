# OmniFind — Streamlit demo UI

A real (not mocked) UI for the backend, used only because the actual Tauri +
React frontend is blocked on installing Node.js + Rust. Talks to the same
FastAPI backend the real frontend will use — no backend changes needed later.

## Run

Streamlit needs its **own** virtual environment, separate from
`omnifind/backend/.venv`. Streamlit's latest version pulls in a newer
`starlette` than the one FastAPI pins in the backend venv, and the two
conflict if installed together (`ImportError: cannot import name
'DEFAULT_EXCLUDED_CONTENT_TYPES'`). Keeping them apart avoids that entirely.

```powershell
# terminal 1 — backend
cd omnifind\backend
.venv\Scripts\activate
uvicorn main:app --reload --port 8000

# terminal 2 — demo UI (first time only: create its own venv)
cd omnifind\streamlit_demo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Next time, just reactivate `omnifind\streamlit_demo\.venv` and run
`streamlit run app.py` — no need to recreate it.

Streamlit opens at `http://localhost:8501`.

## Demo flow for the teacher

1. **Dashboard** — shows it's live, not a mock (file/chunk counts from real SQLite).
2. **Index a folder** — paste a folder path (defaults to `omnifind/examples`), click
   *Start indexing*, watch the live progress bar.
3. **Search** — ask a natural-language question with no literal keyword overlap
   with the file content (e.g. "how much did I pay in fees?" instead of
   "invoice"). Point out the match % and that documents *and* images are
   ranked together in one list.
