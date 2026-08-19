# OmniFind — Streamlit demo UI

> **Superseded.** The real React + TypeScript frontend now lives in
> `omnifind/frontend/` and covers everything here plus the Ask AI page. Use that
> instead; this is kept as a fallback that runs without Node.js installed.

A real (not mocked) UI for the backend, built as a stopgap while the React
frontend was still blocked. Talks to the same FastAPI backend — no backend
changes were needed to move on.

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
