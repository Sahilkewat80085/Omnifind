# OmniFind — Status

AI-based context-aware retrieval system for heterogeneous digital assets.
Semantic search over your own local files, plus cited AI answers drawn from them.

---

## What's built

### Milestone 1 — semantic indexing and retrieval engine ✅

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

### Milestone 2 — RAG answering layer ✅

| # | Module | What it does |
|---|---|---|
| 13 | **LLM provider layer** | `LLMProvider` protocol + Gemini implementation. RagService depends on the protocol, so tests run with a fake and no API key |
| 14 | **RagService** | Retrieves the most relevant passages, builds a numbered excerpt prompt, and asks Gemini to answer **from those excerpts only** |
| 15 | **`POST /ask`** | Returns the answer, the citations behind it, any matching images, and the model used |
| 16 | **Matching images on Ask** | A parallel CLIP pass finds pictures that match the question and returns them as `related_images`, shown beside the answer |

Answers cite inline as `[1]`, `[2]`, and every citation carries the file name,
page number, similarity score and the exact quoted chunk — so any claim can be
traced back to the page it came from.

**Text and images are retrieved separately here, on purpose.** Only text goes
into the prompt: an image embedding carries no words a language model can read,
so an image can never be evidence for a sentence in the answer. Images matching
the question are still retrieved — on their own CLIP pass, scored through the
same calibration the Search page uses — and rendered under a **Matching images**
heading, kept clearly apart from **Sources**. Putting them in the citation list
would claim the model read something it cannot see. The system prompt is told
this too, so it never answers "you have no picture of that" while the matching
picture sits on screen; instead it points the user at the panel.

### Milestone 3 — code search ✅

Point OmniFind at a repository and it indexes source the same way it indexes
documents: ask what a piece of code *does* and get the function back.

| # | Module | What it does |
|---|---|---|
| 17 | **Code parser** | Reads 33 source extensions, preserving exact line structure. Rejects binary content wearing a source extension, and falls back to cp1252 for files from older Windows editors |
| 18 | **Code chunker** | Splits on function/class boundaries so a hit is a whole symbol, not an arbitrary window. Decorators and doc comments stay attached to what they describe |
| 19 | **Scanner pruning** | Never descends into `node_modules`, `.git`, `.venv`, `__pycache__`, `dist`, `site-packages` or any dot-directory |
| 20 | **`CodeResult`** | Search results carry language, symbol name and line range instead of a page number |
| 21 | **Code relevance band** | Code is scored through its own measured floor/ceil and dropped below it, so unrelated functions stay out of results and out of the prompt |

**Code shares the text vector with documents rather than getting a third
model.** Two reasons. Adding a named vector to a live Qdrant collection means
recreating it, throwing away every document and image already indexed. And
unlike an image, code *is* text a language model can read — sharing the
partition is what makes `/ask` cite a function with no extra retrieval path at
all. The payload's `file_type` is what separates them again on the way out.

**What gets embedded is not the bare source.** `bge-small-en` is trained on
English prose, and raw code embeds poorly against a question phrased in words.
Each chunk is given a natural-language header — the file's path within the
indexed folder and the symbol it defines — because those are the terms people
actually search for. The stored snippet stays pure source, so the UI shows real
code.

Two things the document path does that would be actively wrong here: prose
chunking joins on whitespace, which erases the indentation that *is* the syntax
in Python and YAML; and a document is located by page, where code needs a line
range that an editor can open.

**Code needs its own relevance band, and this is the subtle part.** Sharing the
embedding model does not mean sharing the threshold. Two things push code's
noise level far above a document's: every chunk carries an English header
naming its path and symbol, which lifts its similarity to *any* English query,
and one source file yields dozens of chunks, so it gets dozens of chances to
surface. Scored through the document floor of 0.40, bge's noise range came back
as a confident-looking 48-55% match — a lane-detection module ranked against
"a photo of mountains".

Measured against an indexed 1100-line perception module:

| | raw bge score |
|---|---|
| Worst false positive (unrelated query) | 0.593 |
| Weakest genuine match | 0.719 |
| Best genuine match | 0.881 |

The gap between 0.60 and 0.72 is empty, so `search_code_score_floor` sits
inside it at 0.63. Code is **dropped** below its floor rather than clamped —
like images, unlike documents — because an unrelated function in a result list
is a wrong answer, not a weak one. The same floor gates RAG retrieval, so noise
cannot reach the prompt either.

### Query type intent ✅

Naming a file type in the query filters to it. "mountain image" searches
pictures only; "invoice pdf" searches documents; "kalman filter code" searches
source. A brochure that discusses mountains is a wrong answer to a search for a
mountain *image*, however strong the semantic match — the user named the type
they wanted.

The type word is also stripped before embedding: "image" describes the
container, not the content, and leaving it in only blurs the vector the subject
is matched against. Stranded stopwords go with it, so "picture of a dog"
embeds as "dog".

Three things keep this from becoming a source of silent wrong answers:

- **Ambiguity filters nothing.** "image processing code" names two types and
  means neither as a filter, so everything is searched — the old behaviour,
  which is never wrong, only unhelpful.
- **The vocabulary is deliberately narrow.** A word qualifies only if it is
  almost never the subject of a search itself: "photo" is a file type,
  "portrait" is a subject. `class` and `method` are excluded outright — in a
  student's own files, "class notes" is far likelier than either sense meant
  here.
- **The filter is shown, never silent.** The results header reads "3 results …
  images only", and an empty result says which category was searched and how to
  drop the restriction. A filter the user cannot see is indistinguishable from
  broken search.

### Milestone 11 — Desktop UI ✅

A full React + TypeScript + Vite frontend in `frontend/`: Dashboard, Index
folder, Search, Ask AI and Settings, in a dark professional theme.

Node v24 is installed and the whole stack has been run end-to-end — backend on
:8000, Vite dev server on :5173, with real indexing, search and cited answers.
See [frontend/README.md](frontend/README.md).

Both Search and Ask show image thumbnails through the backend's
`/files/{id}/raw` endpoint rather than the file's own path: a page served over
http cannot load a local `file://` URL, so the bytes have to be streamed by the
API. Serving them by **file id** — resolved through SQLite — means only files
the user actually indexed are reachable, instead of anything on disk.

---

## Running it

OmniFind needs the internet **once**, during setup, to download its embedding
models (~740 MB). After `fetch_models.py` has run, every part of the app —
indexing, semantic search, the folder watcher — works with the connection off.

### As a desktop app

Double-click **`OmniFind.bat`**, or:

```powershell
cd omnifind\backend
.venv\Scripts\python.exe desktop.py
```

A native window opens — no browser, no URL to visit. The backend runs inside
that same process on an OS-assigned port and stops when the window closes, so
there is no server left running afterwards and nothing to start by hand.

This needs the frontend built once (`cd omnifind\frontend && npm run build`);
the backend serves the bundle itself, which is also why the UI and the API
share one origin.

### As a web app (development)

Two terminals, with Vite's hot reload:

```powershell
# terminal 1 — backend
cd omnifind\backend
.venv\Scripts\activate
python scripts\fetch_models.py   # one-time, needs internet
uvicorn main:app --reload --port 8000

# terminal 2 — frontend (one-time: winget install OpenJS.NodeJS.LTS, then reopen terminal)
cd omnifind\frontend
npm install
npm run dev        # http://localhost:5173
```

For AI answers, add a Gemini key to `backend/.env`:

```
GEMINI_API_KEY=your-key-here
```

Get one at https://aistudio.google.com/apikey. Search works fine without it —
only the Ask AI page needs it, and the UI says so plainly when it's missing.

### Alternative demo UIs

- **Swagger** — `http://127.0.0.1:8000/docs`, every endpoint testable in-page.
- **Streamlit** — `streamlit_demo/`, the stopgap UI built before the React one.
  Still works; the React app supersedes it.

---

## Tests

```powershell
cd omnifind\backend
.venv\Scripts\activate
pytest -v
```

**70 tests, all passing** — scanning, parsing, chunking, both embedding models,
vector storage, full index→search integration, cross-modal score calibration,
the complete RAG path (prompt construction, citation numbering, relevance
filtering, no-context short-circuit, missing-key handling, and the image pass:
matching pictures are returned, sub-noise ones dropped, and none of them ever
reach the prompt), and code indexing (dependency directories are never walked
even when nested, symbol boundaries hold, decorators stay attached, reported
line numbers reproduce the chunk exactly, and a function is found from a
plain-language question that shares none of its words, and noise-level code
is dropped while the identical score survives as prose), and query type
intent (a named type filters to it, ambiguity filters nothing, and content
words like "class notes" are never mistaken for a type).

---

## Demo script

1. **Dashboard** — live file/chunk counts straight from SQLite, plus recently
   indexed files. Nothing mocked.
2. **Index folder** — point it at `omnifind/examples` (or any real folder) and
   watch the live progress bar.
3. **Search** — the core thesis. Ask something with *no keyword overlap* with
   the target file, e.g. `"how much money was paid"` against an invoice.
   Windows Search cannot do this; OmniFind ranks it first. Note that documents
   and images come back in one merged list despite using two different models.
4. **Ask AI** — ask a question in full sentences, e.g. `"what was billed on the
   invoice?"`. Point out that the answer cites `[1]`, and the Sources panel
   below shows the exact chunk and page each citation came from — this is what
   separates RAG from a chatbot guessing.
5. **Ask AI, visually** — now ask `"is there a picture of a dog in my files?"`.
   The answer says the text gives it nothing to go on, and a **Matching images**
   panel appears below with the photo. Worth calling out: the invoice question
   in step 4 returns *no* images at all. The picture is not padding on every
   answer — it appears only when it actually matches.
6. **Index a codebase** — point step 2 at `omnifind/backend` instead. It
   finishes in seconds, having walked past `.venv` entirely; without that
   pruning it would try to index the whole of scikit-learn.
7. **Say the file type you want** — search `"picture of a dog"`. Only images
   are searched, and the results header says "images only". Then search
   `"mountain image"`: it returns nothing and explains that only images were
   searched — which is correct, because the brochure that mentions mountains is
   a PDF, not a picture. Naming the type is what stops a document answering a
   question about a photo.
8. **Search the code** — ask `"where do we calibrate scores across the two
   modalities?"`. The `_calibrate` function comes back first at ~91%, with its
   real line numbers, even though the query names neither the function nor the
   file. Then ask `"what stops indexing from walking into node_modules?"` — the
   scanner, at ~83%.
9. **Ask about the code** — `"how does the frontend remember which backend URL
   to talk to?"` returns a cited answer pointing at `SettingsPage.tsx` with line
   ranges. This is the same RAG path as the documents, with no extra retrieval
   code: code is text the model can read, so it flows through unchanged.

### Talking points

- **"It understands meaning, not filenames."** The no-keyword-overlap query is
  the single strongest proof.
- **"Two data types, one searchable space."** Text and images use different
  models (bge-small vs. OpenCLIP) but share one Qdrant collection.
- **"The AI cannot make things up about your files."** The prompt forbids
  outside knowledge, and when retrieval finds nothing the model is never
  called — the app says so instead of inventing an answer.
- **"The UI never claims the AI saw more than it did."** Matching images sit
  under their own heading, never in Sources, because the model cannot read
  them — an honest boundary most demos blur.
- **"One engine, three kinds of content."** Documents, images and source code
  go through one index and one ranked list. Code rides the text vector rather
  than a third model, which is why adding it did not invalidate a single
  document already indexed.
- **"It searches code the way a developer thinks."** Chunks follow function and
  class boundaries, so a result is a whole symbol with the line range to open
  it — not a 512-character window sliced through the middle of a function.
- **"It respects what you asked for, not just what you meant."** Say
  "image" and it searches images — and tells you it did, rather than filtering
  in secret.
- **"Every modality is calibrated against measurements, not guesses."** Text,
  images and code each have their own floor, each taken from real score
  distributions. Sharing an embedding model does not mean sharing a threshold.
- **"Production-shaped, not a script."** Clean layering (api → services →
  core), typed config, structured logging, 38 automated tests.

---

## Still ahead

| Item | Status |
|---|---|
| **Tauri desktop wrapper** | Needs Rust + MSVC build tools. Additive only — the React app and API are unchanged by it. Steps in [frontend/README.md](frontend/README.md). |
| OCR, image captioning, face/object detection | Future milestone |
| Knowledge graph, hybrid retrieval, metadata search | Future milestone |
| Continuous folder monitoring, cloud storage | Future milestone |
