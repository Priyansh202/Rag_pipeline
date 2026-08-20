# How this RAG chatbot works

This document is the full picture of the project: what it does, what we use, how a question flows through the system, and what each file is for.

---

## What this project is

A **secure multi-source question-answering chatbot**.

1. You sign in with a **role** (Manager, Assistant Manager, or Developer).
2. You upload **PDFs** and/or index **websites**.
3. You ask questions in a chat UI.
4. The system **searches only the sources your role is allowed to see**, then an LLM writes an answer with **citations** (PDF page or URL).
5. Follow-up questions use **short chat memory** (last 8 turns), so “how many units?” still knows you meant Alta Merita.

It is **not** a general internet search engine. It only answers from **indexed** PDFs and websites.

---

## What we use (the stack)

| Piece | What we use | Why |
|---|---|---|
| UI | Streamlit (`frontend/app.py`) | Chat, login, upload PDFs, add URLs |
| API | FastAPI + Uvicorn | Auth, ingest, query, streaming answers |
| Login | JWT (`PyJWT`) | Username + **role** inside the token |
| Catalog DB | PostgreSQL | Document list (`documents`) + **chunk text for BM25** (`chunk_texts`) |
| Vector DB | **Pinecone** (production / Railway) | Stores embeddings so we can search by meaning |
| Local fallback | **pgvector** in Docker Postgres | Same idea as Pinecone, for local/dev without Pinecone |
| Embeddings | **Voyage AI** `voyage-3-lite` (512-d) | Turns text into vectors. Low RAM (good for Railway) |
| Local embeddings extra | `sentence-transformers` + torch | Optional; heavy RAM; not used on Railway |
| LLM answers | **OpenRouter** (`openai/gpt-4o-mini` by default) | Writes the final answer from retrieved chunks |
| PDF text | PyMuPDF | Extracts text **per page** so citations can say `p. 45` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits pages into ~1000-character chunks |
| Web ingest | httpx → Playwright → optional ScraperAPI | Scrape public pages; block private/local URLs (SSRF) |
| Hybrid search | Vector similarity + **full-corpus BM25** | Meaning search + keyword match over **all** indexed chunks (Postgres `chunk_texts`) |
| Deploy | Railway (API + UI services) + Railway Postgres + volume `/data` | Cloud run |
| Local DBs | Docker Compose `pgvector/pgvector:pg16` | Postgres on port **5433** |

Secrets live in `.env` locally and in Railway variables. Do not commit `.env`.

---

## Roles (who can see what)

| Role | PDFs | Websites |
|---|---|---|
| Manager | yes | yes |
| Assistant Manager | yes | no |
| Developer | no | yes |

Enforced in **three** places so one missed check cannot leak data:

1. **API** — upload/list/delete routes require PDF or web permission.
2. **Agent** — only allowed tools run (`pdf_search` / `web_search`).
3. **Index** — Pinecone/pgvector queries filter `source_type = pdf` or `web`.

---

## End-to-end: what happens when you ask a question

Example: *“What is Alta Merita?”* then *“How many units?”*

```
Streamlit UI
    → POST /query/stream  (JWT + question + last 8 chat turns)
        → sanitize question
        → expand search query with prior user questions  (memory)
        → pick tools (PDF only if the question names an indexed PDF)
        → Voyage embed the search query
        → Pinecone: top candidate chunks (RETRIEVAL_CANDIDATE_K, e.g. 40)
        → Postgres chunk_texts: BM25 top candidates over **entire corpus**
        → merge vector + BM25, re-score, skip table-of-contents junk pages
        → keep top RETRIEVAL_K chunks (e.g. 8–20), full chunk text
        → OpenRouter: history + chunks → streamed answer
    → UI shows answer + citation dropdown
```

**Indexing a PDF** (separate path):

```
Upload PDF
    → save under data/pdfs/
    → extract text per page
    → split into ~1000-char chunks (200 overlap)
    → Voyage embed all new chunks (batch ~110, no delay on paid key)
    → upsert into Pinecone (vectors + metadata)
    → upsert chunk text into Postgres `chunk_texts` (for full-corpus BM25)
    → insert a row in Postgres `documents` catalog
```

---

## Auth in detail

There is **no password database**. “Login” means: pick a username + role, get a **JWT**, send that JWT on every later request.

### 1. Issue a token — `POST /auth/login`

File: `app/api/auth.py`

Body:

```json
{ "username": "demo.user", "role": "manager" }
```

Steps:

1. Username is sanitized (`app/security.py`) — trimmed, max 64 chars, must not be empty.
2. `create_access_token` in `app/auth/jwt_handler.py` builds a JWT with:
   - `sub` = username
   - `role` = `manager` | `assistant_manager` | `developer`
   - `iat` / `exp` = issued at / expiry (`JWT_EXPIRE_MINUTES`, default 120)
3. Token is signed with `JWT_SECRET` using **HS256**.
4. Response includes `access_token`, `role`, `expires_in`, and `allowed_tools` (so the UI knows which buttons to show).

Anyone who knows the API can mint a token for any role. This is a **demo RBAC gate**, not production identity (no password, no user table).

### 2. Every protected route reads the Bearer token

File: `app/auth/dependencies.py`

The UI sends:

```
Authorization: Bearer <jwt>
```

`get_current_user`:

1. Requires the `Bearer` scheme. Missing token → **401**.
2. `decode_access_token` verifies signature and expiry. Bad/expired token → **401**.
3. Reads `sub` and `role`. Unknown role → **403**.
4. Returns a `CurrentUser(username, role)` object.

### 3. Role matrix — `app/auth/rbac.py`

| Role | Tools | Can upload/search PDFs | Can add/search websites |
|---|---|---|---|
| `manager` | `pdf_search`, `web_search` | yes | yes |
| `assistant_manager` | `pdf_search` | yes | no |
| `developer` | `web_search` | no | yes |

### 4. Three enforcement layers (why a Developer cannot see a handbook)

**Layer A — API routes**

- PDF list/upload/delete: `Depends(require_pdf_access)` in `app/api/documents.py`
- Website list/add/delete: `Depends(require_web_access)` in `app/api/websites.py`
- Query: only `get_current_user` (any signed-in role may ask). The **agent** then restricts *which indexes* are searched.

Developer calling `POST /documents` → **403** before ingest runs.

**Layer B — agent tools**

`tools_for_role(role)` in the orchestrator. Developer never calls `search_pdfs()`. Assistant Manager never calls `search_web()`.

**Layer C — vector filter**

`hybrid_search("pdf", …)` vs `hybrid_search("web", …)` always passes `source_type` into Pinecone:

```text
filter: { "source_type": { "$eq": "pdf" } }   // or "web"
```

Even if someone mixed tools, PDF vectors are not returned from a web query.

Query route itself is allowed for all roles so a Developer can still ask about indexed sites. They simply get **zero PDF chunks**.

---

## Document indexing in detail (PDFs)

Entry: UI **Index PDF** → `POST /documents` (`app/api/documents.py`).

The route checks PDF permission, rejects non-`.pdf`, rejects files over `MAX_UPLOAD_MB`, then runs `ingest_pdf` in a **thread** (`asyncio.to_thread`) so Uvicorn is not blocked for minutes.

### Step-by-step — `app/retrieval/pdf_processor.py`

**1. Safe name + whole-file hash**

- Filename cleaned (`sanitize_filename`) — no path tricks, `.pdf` only.
- SHA-256 of the raw bytes. If that exact file is already in Postgres and the disk path still exists → **return existing record** (no re-embed).

**2. Write a temp file** under `data/pdfs/_tmp_{id}_{name}.pdf`.

**3. Extract text per page (PyMuPDF)** — `extract_pdf_documents`

For each page:

- `page.get_text("text")`
- Collapse whitespace
- Skip pages with fewer than **40** characters (blank/image-only pages)
- Split that page’s text with LangChain `RecursiveCharacterTextSplitter`:
  - `CHUNK_SIZE` ≈ 1000 characters
  - `CHUNK_OVERLAP` ≈ 200
  - separators: paragraph → newline → sentence → space → characters

Each chunk is a LangChain `Document` with metadata:

| Field | Meaning |
|---|---|
| `source_type` | `"pdf"` |
| `source_id` | document id |
| `title` | filename |
| `page` | **1-based page number** (this is why citations say `p. 45`) |
| `chunk_index` | order on that page |

If the PDF has no extractable text → error (scanned PDFs without OCR fail here).

**4. Chunk hashes — `app/retrieval/dedup.py`**

Each chunk text is hashed. If a **new** upload overlaps an existing PDF above `pdf_update_overlap_ratio` (default 0.55), we treat it as an **update** of that document:

- Keep the same `source_id`
- `sync_documents`: embed only new/changed chunks, delete removed ones, reuse vectors for unchanged hashes

Otherwise it is a **new** document with a new UUID.

**5. Embed — `app/retrieval/embeddings.py` (`VoyageEmbeddings`)**

All chunk strings go to Voyage `POST https://api.voyageai.com/v1/embeddings`:

- model `voyage-3-lite`
- `input_type=document`
- batch size from `VOYAGE_EMBED_BATCH_SIZE` (capped at 128)
- delay `VOYAGE_BATCH_DELAY_S` between batches (0 on paid)
- HTTP 429 → exponential backoff / `Retry-After`

Output: one **512-d** vector per chunk.

**6. Store vectors — `app/retrieval/vectorstore.py` → `add_documents("pdf", …)`**

Each Pinecone record:

- **id:** `{source_id}:{content_hash}`
- **values:** embedding
- **metadata:** `source_type`, `source_id`, `title`, `page`, `chunk_index`, **full chunk `content`** (up to 8000 chars in metadata)

Upsert in batches of 100 (`app/db/pinecone_store.py`).

Local fallback: same rows in Postgres `chunks` with a `vector` column (pgvector).

**6b. Chunk text catalog — `app/db/chunk_catalog.py`**

Every index/update/delete also syncs Postgres table **`chunk_texts`**:

- **id:** `{source_id}:{content_hash}` (same as Pinecone id)
- **source_type**, **source_id**, **title**, **page**, **url**, **chunk_index**
- **content:** full chunk text (used for BM25 over the **whole** indexed corpus)

BM25 does **not** read from Pinecone at query time. It loads all rows for `pdf` or `web` from `chunk_texts`, scores them, and returns the top keyword hits. On startup, `backfill_chunk_texts()` copies text from Pinecone metadata or pgvector if `chunk_texts` was empty (e.g. after upgrading an existing deploy).

**7. Catalog row — `app/retrieval/registry.py`**

Postgres table `documents`:

- `id`, `kind='pdf'`, `title`, `pages`, `chunks`, `status='ready'`
- `extra` JSON: disk path, `content_hash`

The sidebar list comes from this table, **not** from Pinecone.

**8. Disk**

Final file: `data/pdfs/{source_id}_{filename}.pdf` (Railway: under `DATA_DIR=/data`).

---

## Web indexing and web retrieval in detail

“Web” here means **pages you chose to index**, not live Google search.

### Indexing a URL — `POST /websites`

File: `app/api/websites.py` → `ingest_website` in `app/retrieval/web_scraper.py`.

**1. SSRF guard — `validate_public_http_url`**

Rejects:

- non-http(s)
- localhost / `.local`
- DNS that resolves to private, loopback, link-local, reserved, or multicast IPs

So the scraper cannot hit `http://127.0.0.1:5433` or internal cloud metadata.

**2. Fetch HTML (first success wins)**

Order:

1. **ScraperAPI** if `SCRAPERAPI_KEY` is set (`render=true` for JS)
2. **httpx** GET with a random desktop User-Agent
3. **Playwright Chromium** (headless), wait ~2.5s after `domcontentloaded` for Cloudflare/JS

A fetch is treated as failed if:

- status 401/403/429/503
- body looks like a Cloudflare challenge (“just a moment”, “checking your browser”, …)
- extracted text is fewer than 80 characters

**3. HTML → text**

BeautifulSoup (`lxml`):

- Strip `script`, `style`, `nav`, `footer`, `iframe`, …
- Title from `<title>`
- Visible text, whitespace cleaned

**4. Cache**

`data/web_cache/{hash16}.txt` stores title, URL, body. Re-index of the same URL hash can skip a re-scrape if cache exists.

**5. Chunk + embed + store**

Same splitter as PDFs (`CHUNK_SIZE` / overlap). Metadata:

- `source_type="web"`
- `url` = final URL
- **no page number** — citations use the URL as `locator`

Then `add_documents("web", documents)` → Voyage → Pinecone with `source_type=web`, plus **`chunk_texts`** in Postgres.

Postgres catalog row: `kind='web'`, `url`, chunk count.

### Retrieving web content at question time

There is **no second scrape** on ask.

`search_web(query)` in `app/agent/tools/web_tool.py`:

1. `hybrid_search("web", query, k=RETRIEVAL_K)`
2. Pinecone query **filtered to `source_type=web` only** (vector candidates)
3. Postgres **`chunk_texts`**: BM25 over all web chunks (full corpus)
4. Merge vector + BM25 candidates, re-rank, skip low-content / TOC-like chunks
5. Citations: `locator = url`, `snippet = full chunk text`

Developer questions only run this path. Manager may run PDF and web **in parallel** (thread pool) unless the question matches a PDF filename (then PDF only, to avoid unrelated site chunks).

---

## Retrieval in detail (what happens on “What is Alta Merita?”)

Files: `app/api/query.py` → `stream_agent` / `run_agent` → `retrieve_citations` → `hybrid_search`.

### 1. Sanitize + memory

- Question trimmed, max `MAX_QUERY_CHARS` (2000).
- Last 8 chat turns normalized (`app/agent/memory.py`).
- **Search string** = last few **user** questions + current question.

So after “What is Alta Merita?”, “How many units?” searches roughly:

`What is Alta Merita? How many units?`

That is how follow-ups still hit the OM PDF.

### 2. Which tools run

`_select_tools`:

- Start from JWT role’s allowed tools.
- If **both** PDF and web are allowed, and the search string shares tokens (length ≥ 4) with an indexed **PDF title** (e.g. `Alta_Merita_OM_2.pdf` → `alta`, `merita`) → run **`pdf_search` only**.

Otherwise Manager runs both tools **at the same time** (`ThreadPoolExecutor`).

### 3. Embed the query

Voyage `input_type=query` → one 512-d vector. Must match the **same model/dimension** as indexed chunks (`voyage-3-lite` / 512). Mixing MiniLM (384) and Voyage (512) makes old vectors unusable.

### 4. Vector candidates

Pinecone `query`:

- `top_k` = `RETRIEVAL_CANDIDATE_K` (default 40)
- filter `source_type`
- `include_metadata=true` (chunk text lives in metadata)

Distance used later: `1 - cosine_score`.

pgvector path: `ORDER BY embedding <=> query LIMIT k`.

### 5. Hybrid search (`hybrid_search`)

File: `app/retrieval/vectorstore.py` — used by both PDF and web tools.

**Two parallel candidate pools** (size = `RETRIEVAL_CANDIDATE_K`, default 40):

1. **Vector** — Pinecone (or pgvector) top-K by cosine similarity, filtered by `source_type`.
2. **BM25** — top-K from **every chunk** in Postgres `chunk_texts` for that source type (`app/retrieval/bmb25.py` + `app/db/chunk_catalog.py`). IDF is computed over the **full corpus**, not just the vector hits.

**Merge and re-rank:**

1. Union chunk ids from both pools (dedupe by `{source_id}:{content_hash}`).
2. **Vector sim** = `1 / (1 + distance)` for vector hits; `0` if the chunk only came from BM25.
3. **BM25 score** from full-corpus scoring for every merged chunk (normalized min–max).
4. **Combined:** `HYBRID_ALPHA * vector + (1 - HYBRID_ALPHA) * bm25` (default alpha **0.6** → 60% semantic, 40% keyword).
5. **Quality** — table-of-contents pages (short, “the OFFERING / the PROPERTY / the FINANCIALS”, lots of ALL CAPS) get a **penalty** and are deprioritized if better chunks exist (`app/retrieval/chunk_quality.py`).

Keep top **`RETRIEVAL_K`** (default 8 in code; often set higher in `.env`, e.g. 20). Each hit is the **full stored chunk**, not a truncated snippet.

**Why full-corpus BM25 matters:** Previously BM25 only re-ranked the ~40 vector candidates. If the table of contents ranked high in vectors, the real property description on page 5 never entered the pool. Now BM25 can surface page 5 directly from keyword overlap (“Alta Merita”, “units”, etc.) even when vectors favor page 2.

### 6. Citations object

`search_pdfs` / `search_web` wrap hits as `Citation`: title, locator (`p. N` or URL), snippet, score, page/url.

Duplicates dropped; list sorted by score.

---

## Agent orchestration in detail

This is **not** a free-form LangChain agent that loops and calls random tools. It is a **fixed pipeline** that uses LangChain **tools as named retrievers**.

### What “orchestration” means here

File: `app/agent/orchestrator.py`

```
JWT role
  → allowed tool names
  → optional PDF-only if title matches
  → retrieve (pdf and/or web)
  → build LLM messages (system + history + question + chunks)
  → OpenRouter chat/stream
  → if no LLM: extractive fallback (paste top snippets)
```

LangChain `@tool` objects exist (`pdf_search`, `web_search`) so tools are real LangChain tools, but the **orchestrator calls `search_pdfs` / `search_web` directly**. It does **not** let the LLM choose tools via a ReAct loop. That keeps RBAC strict: the model cannot request `pdf_search` if the role forbids it.

(`choose_tools` in `app/agent/llm.py` exists but is **not** used on the query path.)

### Messages sent to the LLM — `app/agent/llm.py`

Provider from `LLM_PROVIDER` (default OpenRouter). `chat` / `chat_stream` POST to `{base_url}/chat/completions` with:

- `Authorization: Bearer <OPENROUTER_API_KEY>`
- OpenRouter extra headers: `HTTP-Referer`, `X-Title`

Message list:

1. **System** — only use retrieved context + history; cite locators; don’t invent pages; resolve follow-ups.
2. **Prior turns** — last 8 user/assistant strings (assistant “Tools used:” suffix stripped).
3. **Current user** — question + numbered passages `[1] type=pdf title=… locator=p. 45` + full chunk text.

Streaming (`POST /query/stream`): SSE events `meta` (sources, tools) → `token` → `done`.

If OpenRouter is down: `_extractive_answer` returns the top snippets with citations, no generation.

### What the LLM is *not* doing

- Not browsing the live web
- Not reading the original PDF file on disk at ask time
- Not seeing chunks the role cannot search
- Not running a multi-step tool loop

It only **writes English** from (a) chat memory and (b) the retrieved chunks.

---

## Config that matters

From `.env` / Railway:

| Variable | Meaning |
|---|---|
| `VECTOR_BACKEND=pinecone` | Use Pinecone instead of pgvector |
| `EMBEDDING_PROVIDER=voyage` | Use Voyage API, not local MiniLM |
| `VOYAGE_EMBED_BATCH_SIZE` | Texts per Voyage call (max 128) |
| `VOYAGE_BATCH_DELAY_S` | Sleep between batches (0 on paid) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | How PDFs are split when **stored** |
| `RETRIEVAL_K` | Final number of chunks sent to the LLM and shown as citations |
| `RETRIEVAL_CANDIDATE_K` | How many vector hits **and** BM25 hits to pull before merging (default 40) |
| `HYBRID_ALPHA` | Blend weight: `alpha × vector + (1-alpha) × BM25` (default **0.6**). Lower → more keyword-heavy |
| `HYBRID_BM25_K1` | BM25 term-frequency saturation (default **1.5**). Higher → repeated words count more |
| `HYBRID_BM25_B` | BM25 length normalization (default **0.75**). Higher → long chunks penalized more |
| `DATABASE_URL` | Railway Postgres (catalog + `chunk_texts`) |
| `API_URL` | UI → API base URL |
| `SKIP_STARTUP_REINDEX=1` | Skip slow re-embed on boot |

**Chunk size vs “800 characters”:**  
Chunks are **stored** at up to 1000 characters. We send the **whole stored chunk** to the LLM. There is no extra 800-char cut anymore.

---

## File-by-file: what each part does

### Root / run / deploy

| File | What it does |
|---|---|
| `run.py` | Starts Docker Postgres, then API (`:8000`) and Streamlit (`:8501`) |
| `scripts/start_api.py` | Uvicorn on `$PORT` or 8000 (Railway API) |
| `scripts/start_ui.py` | Streamlit on `$PORT` or 8501 (Railway UI) |
| `scripts/start_railway.py` | If service name/mode looks like UI, start Streamlit; else API |
| `scripts/bootstrap_db.py` | DB helper for first-time setup |
| `scripts/generate_sample_pdf.py` | Builds the ACME handbook demo PDF |
| `Dockerfile` | Image for Railway (no torch; Voyage embeddings) |
| `docker-compose.yml` | Local Postgres + pgvector |
| `railway.toml` | Build/start command for Railway |
| `Procfile` | `python scripts/start_railway.py` |
| `requirements.txt` | Default deps (FastAPI, Pinecone, Streamlit, …) |
| `requirements-local-embeddings.txt` | Extra: torch + sentence-transformers |
| `.env` / `.env.example` | Secrets and tunables |
| `.streamlit/config.toml` | Streamlit server options |
| `README.md` | Original take-home writeup (some parts still mention old pgvector-only setup) |

### Frontend

| File | What it does |
|---|---|
| `frontend/app.py` | Login, role, JWT, upload PDF, index URL, chat, citations, **sends history** on each query |
| `frontend/session_store.py` | Saves token + chat to `data/ui_session.json` so refresh does not wipe the session |

### App entry and config

| File | What it does |
|---|---|
| `app/main.py` | FastAPI app: CORS, routers, startup (wait for DB, schema, Pinecone, optional reindex), `/health` |
| `app/config.py` | All settings from env: JWT, DBs, Voyage, Pinecone, chunk/retrieval sizes |
| `app/security.py` | Safe filenames, query length, **SSRF guard** (no localhost/private IPs) |
| `app/models/schemas.py` | Pydantic models: Role, JWT response, QueryRequest (**question + history**), Citation, documents |

### Auth

| File | What it does |
|---|---|
| `app/auth/jwt_handler.py` | Create/decode JWT (`sub`, `role`, expiry) |
| `app/auth/rbac.py` | Role → allowed tools |
| `app/auth/dependencies.py` | FastAPI deps: current user, require PDF access, require web access |
| `app/api/auth.py` | `POST /auth/login` → JWT |

### API routes

| File | What it does |
|---|---|
| `app/api/query.py` | `POST /query` and `/query/stream` — sanitize, run agent with history, SSE tokens |
| `app/api/documents.py` | List/upload/delete PDFs (PDF roles only); ingest in a thread so the server stays responsive |
| `app/api/websites.py` | List/add/delete websites (web roles only) |

### Agent (chat + tools)

| File | What it does |
|---|---|
| `app/agent/orchestrator.py` | Main loop: pick tools, retrieve, format context, call LLM, stream tokens. Skips TOC-like pages via search ranking. If the question matches a PDF title, **does not search websites**. |
| `app/agent/memory.py` | Last N chat turns; blends prior **user** questions into the **search** string |
| `app/agent/llm.py` | OpenRouter / OpenAI / Groq / Ollama / none; chat + stream |
| `app/agent/tools/pdf_tool.py` | `search_pdfs` — hybrid search on `source_type=pdf`, returns full chunks |
| `app/agent/tools/web_tool.py` | Same for websites |

### Retrieval (index + search)

| File | What it does |
|---|---|
| `app/retrieval/pdf_processor.py` | Save PDF, extract pages, chunk, dedup/update, embed + store |
| `app/retrieval/web_scraper.py` | Fetch page text (ScraperAPI / httpx / Playwright), chunk, index |
| `app/retrieval/embeddings.py` | Voyage HTTP embeddings (batch, 429 retry) or local MiniLM |
| `app/retrieval/vectorstore.py` | Add/sync/delete/search chunks; Pinecone or pgvector; **hybrid** merge (vector + full-corpus BM25) + TOC downrank |
| `app/retrieval/chunk_quality.py` | Detect table-of-contents / heading-only chunks so they don’t beat real pages |
| `app/retrieval/bmb25.py` | BM25 keyword scoring (tokenize, IDF, score) over a chunk list or full corpus |
| `app/retrieval/registry.py` | Postgres catalog CRUD (the document list in the sidebar) |
| `app/retrieval/hashing.py` | Hash of chunk text |
| `app/retrieval/dedup.py` | If a new PDF overlaps an old one, **update** instead of duplicating |
| `app/retrieval/bootstrap.py` | On startup: migrate old `registry.json`, reindex missing vectors, **backfill `chunk_texts`** |

### Databases

| File | What it does |
|---|---|
| `app/db/postgres.py` | Pool + `documents` table (catalog) + **`chunk_texts`** table (BM25 corpus) |
| `app/db/chunk_catalog.py` | Upsert/delete/fetch chunk text; **top_bm25_for_kind** over full corpus |
| `app/db/pinecone_store.py` | Create index, upsert, query by `source_type`, fetch metadata for backfill, list/delete by source |
| `app/db/pgvector_store.py` | Local vector table + pgvector extension check |
| `app/db/wait.py` | Wait until Postgres is reachable before serving traffic |

### Tests

| File | What it does |
|---|---|
| `tests/conftest.py` | Force local/test env (no real Voyage/Pinecone keys leaking into tests) |
| `tests/test_rbac.py` | Roles, 401, developer cannot upload PDF, SSE stream |
| `tests/test_security.py` | Filename + SSRF sanitization |
| `tests/test_memory.py` | Follow-up search query includes prior question |
| `tests/test_chunk_quality.py` | TOC page vs real property description |
| `tests/test_bm25.py` | Keyword ranking + full-corpus BM25 top-K |
| `tests/test_hybrid_search.py` | Vector + BM25 merge (BM25 can surface chunks vectors missed) |
| `tests/test_voyage_embeddings.py` | 429 retry |
| `tests/test_dsn.py` | `postgres://` → `postgresql://` |
| `tests/test_dedup.py` | PDF update overlap |
| `tests/test_pdf_e2e.py` | Upload sample handbook and cite page 1 |

### Data on disk

| Path | What it is |
|---|---|
| `data/pdfs/` | Uploaded PDFs (`{id}_{filename}.pdf`) |
| `data/web_cache/` | Cached scraped text |
| `data/ui_session.json` | Local UI login + chat (not used on Railway the same way) |
| `data/registry.json` | Legacy catalog; Postgres is the real catalog now |

---

## Mental model (three stores)

```
Postgres documents  = “what files exist?”        (titles, pages, chunk counts)
Postgres chunk_texts = “exact words in every chunk” (full text for BM25 keyword search)
Pinecone             = “what do they mean?”       (vectors + chunk text in metadata)
Voyage               = “turn text into vectors”
OpenRouter           = “write an English answer from the chunks”
JWT role             = “which of those chunks am I allowed to search?”
```

On **Railway**, Postgres holds both `documents` and `chunk_texts`. Pinecone holds vectors. You do **not** need pgvector on Railway unless you switch `VECTOR_BACKEND=pgvector`.

Locally with **Docker**, one Postgres container (`127.0.0.1:5433`) runs `rag_app` (catalog + `chunk_texts`) and optionally `rag_vectors` (pgvector when not using Pinecone).

---

## Run it locally

```powershell
cd C:\Users\praka\Downloads\rag
.\.venv\Scripts\Activate.ps1
python run.py
```

- API docs: http://127.0.0.1:8000/docs  
- UI: http://127.0.0.1:8501  
- Health: http://127.0.0.1:8000/health  

After changing Python code, **restart the API** (Ctrl+C, start again). Streamlit often hot-reloads; FastAPI does not unless you started uvicorn with `--reload`.

---

## Railway (short)

Two services from the same repo:

- **API** — `START_MODE=api`, `DATABASE_URL`, JWT, Voyage, Pinecone, `DATA_DIR=/data`, volume mount `/data`, health `/health`
- **UI** — `START_MODE=ui`, `API_URL=https://<api-domain>`, health `/_stcore/health`

Postgres is **catalog + chunk text for BM25**. Vectors are in Pinecone. pgvector is **not** required on Railway.
