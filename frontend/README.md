# OmniFind — frontend (React + TypeScript + Vite)

The real UI for OmniFind. Five screens — Dashboard, Index folder, Search, Ask AI,
Settings — talking to the FastAPI backend over HTTP.

## Prerequisite: Node.js

Node v24 is installed on this machine and the app has been run — nothing to do.
On a fresh machine, install it once:

```powershell
winget install OpenJS.NodeJS.LTS
```

Then **close and reopen the terminal** so `node` and `npm` land on your PATH.
Verify with `node --version` (expect v20 or newer).

On npm 11+, `npm install` prints an `allow-scripts` warning about esbuild's
postinstall. It is safe to ignore — the build works regardless.

## Run

```powershell
# terminal 1 — backend
cd omnifind\backend
.venv\Scripts\activate
uvicorn main:app --reload --port 8000

# terminal 2 — frontend (first time only: npm install)
cd omnifind\frontend
npm install
npm run dev
```

Open http://localhost:5173.

The backend URL defaults to `http://127.0.0.1:8000` and can be changed on the
Settings page (stored in browser localStorage).

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Type-check then produce a production bundle in `dist/` |
| `npm run typecheck` | Type-check only, no output |
| `npm run preview` | Serve the built bundle locally |

## Structure

```
src/
  api/
    types.ts       mirrors backend/models/schemas/*.py — the API contract
    client.ts      typed fetch wrapper, ApiError, backend-URL persistence
  components/      Sidebar, SearchBar, FileCard, CodeCard, AnswerCard,
                   ReferenceCard, RelatedImageCard, StatCard, Banner,
                   EmptyState, LoadingIndicator
  pages/           Dashboard, IndexPage, SearchPage, AskPage, SettingsPage
  hooks/           useHealth — polls /health for connection + AI status
  utils/           format.ts (display helpers), relevance.ts (the 30%
                   presentation floor and one-row-per-file rule)
  styles.css       the whole theme, as CSS custom properties
```

## Notes on six deliberate choices

**A detected type filter is always shown, never applied silently.** When the
query names a file type the backend returns `filtered_to`, and the results
header renders an "images only" chip. An empty result then says which category
was searched and how to drop the restriction, because "No item matched" on its
own looks like broken search rather than an empty category.


**The UI has its own relevance floor, separate from retrieval's.**
`utils/relevance.ts` hides anything under 30% and collapses a file to a single
row, keeping its best-scoring passage. Both are presentation rules, not search
ones: the backend deliberately keeps weak text hits for recall, but a 12% row
reads as an answer simply by being listed, and a long file matching on six
chunks used to fill the screen with one repeated filename. When nothing clears
the floor the page says **No item matched** rather than showing a list of
near-misses.


**Code results get their own card.** `CodeCard` renders into a `<pre>` that
preserves whitespace, because indentation is the syntax in Python and YAML —
the prose snippet in `FileCard` collapses it and would display something that
is not the code in the file. It shows a line range rather than a page number,
and `ReferenceCard` does the same for a code citation on the Ask page.


**Image thumbnails load from the API, not from disk.** Both `FileCard` (Search)
and `RelatedImageCard` (Ask) point their `<img>` at
`api.fileContentUrl(file_id)` → `GET /files/{id}/raw`. A page served over http
cannot load a local `file://` path, so the result's own `path` field will never
render as an image — it is there for the "Open" button, which goes through the
OS. Both cards also handle `onError`, since a file can be indexed and later
moved or deleted.

On the Ask page these images render under **Matching images**, deliberately
separate from **Sources**: they were matched visually by CLIP and never entered
the prompt, so presenting them as citations would overstate what the model read.


**No router.** Five flat screens, no deep links or URL parameters, so navigation
is component state. This also sidesteps the `file://` base-path problem that
routers hit once the same bundle is wrapped in Tauri.

**No CSS framework.** The theme is ~200 lines of plain CSS driven by custom
properties in `:root`. Nothing to configure, nothing to purge, and the palette
is one edit away from being restyled.

## Path to the Tauri desktop build

This is the same app the desktop build will ship — the wrapper is additive, and
no application code changes:

1. `winget install Rustlang.Rustup` and the MSVC C++ build tools
2. `npm install -D @tauri-apps/cli` then `npx tauri init`, pointing it at
   `dist/` as the build output and `http://localhost:5173` as the dev server
3. `npm run tauri dev`

`vite.config.ts` already sets `base: "./"` so the built bundle loads correctly
from the filesystem, which is what Tauri needs.
