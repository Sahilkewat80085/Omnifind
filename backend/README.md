# OmniFind Backend

Semantic indexing and retrieval engine for OmniFind. FastAPI + SQLite (file
metadata) + Qdrant (vectors, embedded local mode — no server to run).

No LLM, no RAG, no chatbot in this milestone — pure vector search over
documents (PDF/DOCX/TXT) and images (PNG/JPG/JPEG).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt # or requirements-dev.txt to include pytest

copy .env.example .env          # adjust if needed, defaults work out of the box
```

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
│   └── vectorstore/         Qdrant client wrapper (VectorService)
├── services/                IndexingService, SearchService, MetadataService, IndexJobManager
├── database/                SQLAlchemy models + session
├── models/schemas/          Pydantic DTOs
├── utils/                   Config (env-driven settings), logger
├── storage/                 SQLite DB + local Qdrant data (gitignored)
└── tests/                   pytest suite
```

## Out of scope for this milestone

OCR, LLM/RAG/chatbot, face/object detection, knowledge graph, cloud storage,
authentication, continuous folder monitoring. See project spec for the full
roadmap.
