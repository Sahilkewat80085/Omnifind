# OmniFind Backend

Semantic indexing and retrieval engine for OmniFind. FastAPI + SQLite (file
metadata) + Qdrant (vectors, embedded local mode — no server to run), with a
retrieval-augmented answering layer on top (Gemini).

Vector search over documents (PDF/DOCX/TXT) and images (PNG/JPG/JPEG), plus
cited natural-language answers via `POST /ask`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt # or requirements-dev.txt to include pytest

copy .env.example .env          # adjust if needed, defaults work out of the box

python scripts/fetch_models.py  # one-time, needs internet (~740 MB)
```

### The one-time model download

`scripts/fetch_models.py` is the **only** part of OmniFind that needs an
internet connection. It pulls the two embedding models into the local
HuggingFace cache:

| Model | Purpose | Size |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | text + code embeddings | 135 MB |
| `CLIP-ViT-B-32-laion2B-s34B-b79K` | image embeddings | 605 MB |

After that the app runs **fully offline**. It does not merely happen to work
without a connection — `utils/offline.py` sets `HF_HUB_OFFLINE=1` before any
model library is imported, so the HuggingFace libraries load from the local
cache and never attempt a request. (The variable is read into a module
constant at import time, which is why `main.py` enforces it above its own
imports; setting it later is silently a no-op.)

The script is safe to re-run and doubles as an install check. If it was never
run, the backend still boots, but `/health` reports
`models_state: "unavailable"`, the UI shows a banner naming the fix, and search
returns `503` with the same message instead of a stack trace about an
unreachable host.

### Enabling AI answers

Search works with no configuration. For `/ask`, put a Gemini API key
(from https://aistudio.google.com/apikey) in `.env`:

```
GEMINI_API_KEY=your-key-here
```

Restart the backend afterwards. Without a key the app still boots and search
is unaffected — only `/ask` responds `503` with an explanatory message, and
`/health` reports `ai_enabled: false` so the UI can say so up front.

## Run

```bash
uvicorn main:app --reload --port 8000
```

Visit `http://127.0.0.1:8000/health` to confirm it's up. Interactive API
docs at `http://127.0.0.1:8000/docs`.

## Test

```bash
pytest
```

## Architecture

Clean architecture, four layers:

```
api/        FastAPI routers — HTTP only, no business logic
services/   Use-case orchestration (IndexingService, SearchService, MetadataService)
core/       Framework-agnostic domain logic (scanner, parsers, chunker, embeddings, vector store)
database/   SQLAlchemy models + session
```

Routers call services, services call core modules. Core modules never import
FastAPI, so the whole engine is testable without an HTTP server.

### Indexing pipeline

```
FolderScanner → DocumentParser/ImageProcessor → TextChunker (docs only)
→ EmbeddingService (bge-small for text, OpenCLIP for images)
→ VectorService.upsert (Qdrant) → MetadataService (SQLite)
```

### Search pipeline

Every query is embedded twice — once with bge (searches the document
partition), once with CLIP's text tower (searches the image partition) —
because CLIP can compare text and images directly but bge-embedded text
isn't comparable to CLIP-embedded images. Each partition's cosine scores are
min-max normalized independently before merging, since the two models'
scores live on different scales.

### RAG pipeline (`/ask`)

```
query → bge query embedding → Qdrant text partition (top rag_top_k)
→ drop hits below RAG_MIN_SIMILARITY → numbered excerpt block
→ Gemini (answer strictly from excerpts, cite [n]) → answer + citations
```

Retrieval here goes straight to `VectorService` rather than through
`SearchService`, for two reasons. Image embeddings carry no text a language
model could read, so the image partition is skipped entirely. And
`SearchService` min-max normalizes scores *within* each modality, which forces
its top hit to 1.0 however weak the match — a relevance threshold applied to
those numbers would mean nothing. `RagService` therefore thresholds on raw
cosine scores.

`RagService` depends on the `LLMProvider` protocol (`core/llm/base.py`), not on
Gemini directly, so tests inject a deterministic fake and the whole RAG path is
covered without an API key or a network call.

### Vector storage

Single Qdrant collection `omnifind_assets` using **named vectors**
(`text_vector`, 384-dim; `image_vector`, 512-dim) so both modalities share
one collection without dimension conflicts. Runs in Qdrant's embedded local
mode (on-disk under `storage/qdrant_local`, no separate server process) by
default — set `QDRANT_MODE=server` in `.env` to point at a real Qdrant
instance instead.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/index/scan` | POST `{"path": "..."}` | Start indexing a folder (background job) |
| `/index/status` | GET | Poll current indexing job progress |
| `/index/stats` | GET | Dashboard stats: file/chunk counts, total size |
| `/search?q=...` | GET | Semantic search across documents and images |
| `/ask` | POST `{"q": "...", "top_k": null}` | Cited RAG answer over the indexed documents |
| `/files` | GET | List indexed files (optional `?file_type=document\|image`) |
| `/files/{file_id}/raw` | GET | Stream an indexed file's bytes (used for image thumbnails) |
| `/files/open` | POST `{"path": "..."}` | Open a file with the OS default application |

## Project structure

```
backend/
├── api/                    FastAPI routers
├── core/
│   ├── scanner/             Recursive folder scan + extension filtering
│   ├── parsers/             PDF/DOCX/TXT text extraction, image reading
│   ├── chunking/            Sliding-window text chunker
│   ├── embeddings/          bge-small (text) + OpenCLIP (image) wrappers
│   ├── llm/                 LLMProvider protocol + Gemini implementation
│   └── vectorstore/         Qdrant client wrapper (VectorService)
├── services/                IndexingService, SearchService, RagService, MetadataService, IndexJobManager
├── database/                SQLAlchemy models + session
├── models/schemas/          Pydantic DTOs
├── utils/                   Config (env-driven settings), logger
├── storage/                 SQLite DB + local Qdrant data (gitignored)
└── tests/                   pytest suite
```

## Out of scope

OCR, face/object detection, knowledge graph, cloud storage, authentication,
continuous folder monitoring. See project spec for the full roadmap.
