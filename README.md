# OmniFind — Milestone 1 Status (20% MVP)

AI-based context-aware retrieval system for heterogeneous digital assets.
This document tracks what's built for the first milestone and how to
demonstrate it.

Per the project scope, this milestone is **the core semantic indexing and
retrieval engine only** — no chatbot, no RAG, no LLM. Those are explicitly
later milestones, not gaps in this one.

---

## ✅ Implemented

Everything below is written, running, and covered by automated tests — not
just scaffolded.

| # | Module | What it does |
|---|---|---|
| 1 | **Project setup** | FastAPI backend, typed env-driven config, structured logging |
| 2 | **Database** | SQLite via SQLAlchemy — tracks every indexed file (name, type, path, size, chunk count, image dimensions) |
| 3 | **Folder Scanner** | Recursively walks a chosen folder, filters to supported types |
| 4 | **Document Parser** | Extracts text from PDF (per-page), DOCX, TXT |
| 5 | **Image Processor** | Reads PNG/JPG/JPEG, validates and extracts dimensions |
| 6 | **Text Chunker** | Splits document text into overlapping chunks for embedding |
| 7 | **Embedding Services** | `BAAI/bge-small-en-v1.5` for text, OpenCLIP ViT-B/32 for images |
| 8 | **Vector Storage** | Qdrant (embedded, on-disk — no server to install), named vectors so text and image embeddings share one collection |
| 9 | **Indexing Service + API** | End-to-end pipeline: scan → parse → chunk → embed → store, with progress reporting |
| 10 | **Search Service + API** | Natural-language query → semantic search across **both** documents and images, ranked together |
| 12 | **Test suite** | 20 automated tests (pytest) covering every module above |

**Supported file types:** PDF, DOCX, TXT, PNG, JPG, JPEG — exactly as scoped.

**Not built (by design, in scope for this milestone):** OCR, LLM answers,
RAG, chatbot, face/object detection, knowledge graph, cloud storage,
authentication, continuous folder monitoring. These are documented future
milestones, not missing work.

## ⏳ Remaining for this milestone

| # | Module | Status |
|---|---|---|
| 11 | **Desktop UI** (Tauri + React) | Not started — blocked on installing Node.js and Rust on the dev machine. The entire backend it needs to call is already built and tested. |

Once Node.js + Rust are installed, the UI (Dashboard, Search, Settings
pages; folder picker, progress bar, ranked result cards) can be built
against the existing, working API — no backend changes needed.

---

## How to demonstrate this to your teacher

There's no graphical UI yet, so the demo runs through FastAPI's built-in
interactive API docs (Swagger UI) — this is a legitimate, professional way
to demo a backend and looks better than raw terminal commands.

### 1. Start the server

```powershell
cd omnifind\backend
.venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

### 2. Open the interactive API docs

Go to **http://127.0.0.1:8000/docs** in a browser. You'll see every
endpoint listed and testable from the page directly.

### 3. Index a real folder

- Expand `POST /index/scan` → "Try it out" → enter a folder path containing
  a mix of PDFs, DOCX, TXT, and images (e.g. a Documents or Downloads
  subfolder) → Execute.
- Expand `GET /index/status` → Execute a few times to show `processed` and
  `total` climbing to completion. This is the live progress feed the UI
  will eventually poll.
- Expand `GET /index/stats` → Execute → shows file/chunk counts, proving
  everything landed in the database.

### 4. Run semantic search — the core deliverable

Expand `GET /search`, try queries like:

- A phrase that **doesn't literally appear** in any filename, e.g.
  `"database normalization"` — should return the right document chunk with
  a similarity score and page number, proving it's matching on *meaning*,
  not keywords.
- An image-flavored query, e.g. `"whiteboard diagram"` or `"forest photo"`
  — should return an image result with its dimensions, proving cross-modal
  (text-to-image) search works.
- A broader query, e.g. `"meeting notes"` — should return documents and
  images ranked together in one list.

This is the single strongest proof point: **type a natural-language
sentence, not a filename, and get back the right file** — that's the whole
thesis of the project (semantic vs. keyword search) demonstrated live.

### 5. (Optional but strong) Show the test suite passing

```powershell
pytest -v
```

20 tests, all green — covers scanning, parsing, chunking, both embedding
models, vector storage, and full index→search integration. Good evidence
this isn't a one-off demo but a working, verified system.

### Talking points while presenting

- **"Unlike Windows Search, this understands meaning."** Show a query with
  no keyword overlap with the target file's content — that's the
  differentiator.
- **"It indexes two completely different data types into one searchable
  space."** Text and images use different AI models (bge-small vs.
  OpenCLIP) but land in the same Qdrant collection and get ranked together.
- **"This is production-shaped, not a script."** Clean architecture
  (API → services → core), typed config via `.env`, structured logging,
  automated tests — point to `backend/README.md` for the architecture
  breakdown if asked.
- **What's next**, if asked: wiring this same API up to the desktop UI
  (Tauri + React), which is scoped and ready to build.
