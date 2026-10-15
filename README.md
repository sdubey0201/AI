# RAG Assistant

A **fully local Retrieval-Augmented Generation (RAG)** pipeline that lets you ask questions
against your own Markdown documents and receive synthesised natural-language answers —
no internet connection, no API key, everything runs on your machine.

It uses:
- **LlamaIndex** for document loading, chunking, and vector indexing
- **`sentence-transformers/all-MiniLM-L6-v2`** (HuggingFace) as a fully local embedding model
- **ChromaDB** as a persistent on-disk vector store
- **Ollama (`llama3.2:latest`)** as the local LLM that generates answers from retrieved context

---

## What This Application Does

```
documents/*.md
      │
      ▼  ingest.py
  Load Markdown files
      │
      ▼
  Split into chunks  (chunk_size=128 tokens, overlap=40)
      │
      ▼
  Embed each chunk   (384-dim float vectors, runs locally)
      │
      ▼
  Store vectors      (chroma_db/ on disk, collection "kubernetes_docs")

      ─────────────────────────────────────

  query.py
      │
      ▼
  Embed your question (same embedding model)
      │
      ▼
  Cosine-similarity search in ChromaDB  (top_k=3)
      │
      ▼
  Filter by MIN_SIMILARITY_SCORE (≥ 0.35)
      │
      ▼
  If no results pass → "No relevant information found"
      │
      ▼  (if results pass)
  Build context string from relevant chunks
      │
      ▼
  Fill RAG_PROMPT_TEMPLATE  {context} + {question}
      │
      ▼
  Ollama llama3.2:latest  (runs locally, context_window=4096)
      │
      ▼
  Print Answer + Sources (file name + similarity score)
```

**Two separate concerns, two separate scripts:**

| Script | Responsibility |
|--------|----------------|
| `ingest.py` | One-time (or on-update) step: read docs → chunk → embed → save to DB |
| `query.py` | Interactive step: ask a question → retrieve chunks → generate answer via Ollama LLM |

All settings (paths, model name, chunk sizes, top-k) live in **`config.py`** — no magic numbers scattered across files.

---

## Project Layout

```
rag-assistant/
├── config.py          ← All configuration constants
├── ingest.py          ← Ingestion pipeline (run once per document update)
├── query.py           ← Interactive retrieval loop
├── rag.py             ← Original single-file prototype (kept for reference)
├── documents/
│   └── kubernetes_architecture.md   ← Knowledge base: Kubernetes architecture, containers, pods, troubleshooting
└── chroma_db/         ← Auto-created by ingest.py; do not edit manually
```

---

## Configuration (`config.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `DOCUMENT_DIR` | `documents/` | Folder scanned for `.md` files |
| `DB_DIR` | `chroma_db/` | On-disk ChromaDB storage |
| `COLLECTION_NAME` | `kubernetes_docs` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local HuggingFace embedding model |
| `CHUNK_SIZE` | `128` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `40` | Token overlap between consecutive chunks |
| `SIMILARITY_TOP_K` | `3` | Number of chunks returned per query |
| `MIN_SIMILARITY_SCORE` | `0.35` | Minimum cosine similarity score — results below this are discarded |
| `RAG_PROMPT_TEMPLATE` | *(multiline string)* | Prompt sent to the LLM; injects `{context}` and `{question}` placeholders |

---

## Prerequisites

- Python 3.10 or later
- The `.venv` virtual environment included in this repo
- **Ollama** installed and running locally with `llama3.2:latest` pulled

### Install Ollama (first time only)

```bash
# macOS
brew install ollama

# Pull the model used by query.py
ollama pull llama3.2:latest

# Start the Ollama server (keep this running in a separate terminal)
ollama serve
```

### Install Python dependencies (first time only)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Start the Application

### Step 1 — Activate the virtual environment

```bash
source .venv/bin/activate
```

### Step 2 — Add your documents

Drop any number of `.md` files into the `documents/` folder.  
The current knowledge base already contains `kubernetes_architecture.md`.

### Step 3 — Run ingestion (once, or whenever documents change)

```bash
python ingest.py
```

Expected output:
```
Loaded documents: 1
Created chunks: 6
--- Chunk 1 ---
Kubernetes manages containerized applications ...
...
Ingestion completed successfully.
Indexed chunks: 6
Database location: /path/to/rag-assistant/chroma_db
```

> Re-running `ingest.py` **rebuilds the collection from scratch** — safe to run whenever you add or change documents.

### Step 4 — Start Ollama (in a separate terminal)

```bash
ollama serve
```

> If Ollama is already running as a background service this step is optional.

### Step 5 — Ask questions interactively

```bash
python query.py
```

Expected output:
```
Indexed chunks available: 6

Ask a question (or type 'exit'): Why can a Kubernetes pod remain in Pending state?

Answer:
A Kubernetes pod can remain in Pending state due to insufficient resources on the
available nodes, scheduling constraints such as node selectors or taints, or
unresolved storage (PVC) issues.

Sources:
1. kubernetes_architecture.md (score: 0.912)
2. kubernetes_architecture.md (score: 0.874)

Ask a question (or type 'exit'): What is the capital of France?

Answer: No relevant information found in the knowledge base.

Ask a question (or type 'exit'): exit
```

Type `exit` or `quit` to stop the query loop.

> Results with a similarity score **below `MIN_SIMILARITY_SCORE` (0.35)** are discarded before
> the prompt is built — the LLM never sees irrelevant context.
>
> If Ollama is not running or the model is not installed, `query.py` prints a clear error
> message rather than crashing silently.

---

## Key Concepts

### Embeddings (384-dim vectors)
Every text chunk is converted into a list of 384 floating-point numbers by the embedding model.
Chunks with similar meaning produce vectors that are geometrically close.
Retrieval = find the stored vectors with the smallest angular distance to your question's vector.

### Cosine similarity score
A score of `1.0` means the chunk is a near-perfect semantic match.
A score near `0.0` means the chunk is unrelated to the question.

### RAG prompt template (`RAG_PROMPT_TEMPLATE`)
Defined in `config.py`, this is the instruction string sent to the LLM.
It injects two placeholders at runtime:
- `{context}` — the concatenated text of all chunks that passed the similarity threshold
- `{question}` — the user's question

The template instructs the LLM to answer **only from the provided context** and to say
*"I don't know based on the available documents."* if the context doesn't contain the answer.
This prevents hallucination.

### Ollama (local LLM)
Ollama runs `llama3.2:latest` entirely on your machine. `query.py` calls
`llm.complete(prompt)` which sends the filled prompt to Ollama's local HTTP server
(default: `http://localhost:11434`) and streams back the generated answer.
No data leaves your machine.

### Minimum similarity threshold (`MIN_SIMILARITY_SCORE`)
After retrieval, `query.py` filters out any result whose score is below `0.35`.
This prevents the assistant from confidently returning irrelevant chunks just because
they are the "least bad" match. If **all** results are below the threshold, the query
loop prints `"No relevant information found in the knowledge base."` and waits for the
next question.

Tune this value in `config.py`:
- **Lower** (e.g. `0.2`) → more results, higher chance of noise
- **Higher** (e.g. `0.5`) → fewer results, higher precision

### Why split into chunks?
Embedding models have a token limit. Smaller chunks also give more precise retrieval —
returning exactly the paragraph that answers the question rather than an entire page.

---

## Extending the Application

| Goal | What to do |
|------|-----------|
| Add more documents | Drop `.md` files into `documents/` and re-run `ingest.py` |
| Support PDF / DOCX | Change `required_exts` in `ingest.py` and install `pypdf` / `docx2txt` |
| Add answer generation | Replace the retriever in `query.py` with `index.as_query_engine(llm=...)` + Ollama or OpenAI |
| Expose as a REST API | Wrap `query.py` logic in a FastAPI `/query` endpoint |
| Use a better model | Change `EMBEDDING_MODEL` in `config.py` to `BAAI/bge-base-en-v1.5` |

---

## Changelog

All notable changes to this project are recorded here in reverse-chronological order.
Format: `[vX.Y] YYYY-MM-DD — Short title` followed by bullet points.

> **Convention for future updates:**
> When you make a change to any `.py` file, add a new entry at the top of this section
> describing what changed and why. Use the version bump guide below:
> - **Patch (v0.0.X)** — bug fixes, config value tweaks, typo corrections
> - **Minor (v0.X.0)** — new feature added, new file, new behaviour without breaking existing usage
> - **Major (vX.0.0)** — breaking change (e.g. DB schema change that requires re-ingestion, renamed constants)

---

### [v0.4.2] — Debug logging added to `query.py`

**Files changed:** `query.py`

- **Added** debug block after retrieval — prints all raw results (rank, score, chunk text)
  before the similarity filter is applied; labelled `"Debug: Retrieved results before filtering"`
- **Added** threshold diagnostics — prints `MIN_SIMILARITY_SCORE`, total results before
  filtering, and total results after filtering on every query
- **Purpose:** makes it easy to inspect why results are accepted or rejected during
  development; helps tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `MIN_SIMILARITY_SCORE`
- **Note:** this is development/learning instrumentation — remove or guard behind a
  `DEBUG` flag before using in production

---

### [v0.4.1] — Knowledge base expanded with Containers and Pods sections

**Files changed:** `documents/kubernetes_architecture.md`

- **Added** `## Containers` section — explains containers as lightweight isolated
  environments that package application code, libraries, and runtime dependencies
- **Added** `## Pods` section — explains Pods as the smallest deployable unit in
  Kubernetes; covers multi-container pods, shared network, and shared storage volumes
- **Removed** the old `## Troubleshooting` inline gap (blank line cleaned up)
- **Effect:** the knowledge base now covers 5 topics (Architecture, Control Plane,
  Worker Nodes, Containers, Pods, Troubleshooting); re-run `ingest.py` to index
  the new content before querying

---

### [v0.4.0] — LLM answer generation via Ollama (full RAG pipeline)

**Files changed:** `query.py`, `config.py`

- **Added** `RAG_PROMPT_TEMPLATE` to `config.py` — multiline prompt string with
  `{context}` and `{question}` placeholders; instructs the LLM to answer only from
  provided context and avoid hallucination
- **Added** Ollama LLM integration to `query.py` using `llama_index.llms.ollama.Ollama`
  with model `llama3.2:latest`, `request_timeout=120.0`, `context_window=4096`
- **Added** context assembly step — concatenates all passing chunk texts with double
  newlines before injecting into the prompt
- **Added** answer generation step — calls `llm.complete(prompt)` and prints the
  synthesised answer
- **Added** source attribution — after the answer, prints each source file name and
  similarity score (e.g. `1. kubernetes_architecture.md (score: 0.912)`)
- **Added** graceful error handling — if Ollama is not running or the model is missing,
  prints a helpful message instead of raising an unhandled exception
- **Removed** raw chunk printing — `query.py` no longer prints raw retrieved text;
  the LLM answer replaces it
- **Effect:** the application is now a **complete RAG pipeline** — retrieval + generation.
  Users receive a synthesised natural-language answer, not raw document fragments.
- **Dependency added:** `llama-index-llms-ollama` (install with pip); Ollama must be
  running locally with `llama3.2:latest` pulled

---

### [v0.3.0] — Similarity threshold filtering in `query.py`

**Files changed:** `query.py`, `config.py`

- **Added** `MIN_SIMILARITY_SCORE = 0.35` constant to `config.py`
- **Added** post-retrieval filter in `query.py` — results with a cosine similarity score
  below `MIN_SIMILARITY_SCORE` are discarded before printing
- **Added** graceful message `"No relevant information found in the knowledge base."`
  when all retrieved results fall below the threshold
- **Effect:** prevents the assistant from confidently returning irrelevant chunks when
  the question is outside the scope of the indexed documents

---

### [v0.2.0] — Refactor: separated ingestion from querying, centralised config

**Files changed:** `config.py` (new), `ingest.py` (new), `query.py` (new), `rag.py` (kept as reference)

- **Added** `config.py` — single source of truth for all constants; removed all
  hardcoded values from individual scripts
- **Added** `ingest.py` — standalone ingestion pipeline that rebuilds the ChromaDB
  collection from scratch on each run; prints all chunks to stdout for inspection
- **Added** `query.py` — interactive question loop; connects to the existing DB without
  re-embedding; accepts `exit` / `quit` to stop
- **Changed** `CHUNK_SIZE` from `40` → `128` tokens; `CHUNK_OVERLAP` from `5` → `40` tokens
  for better semantic coherence per chunk
- **Kept** `rag.py` as the original single-file prototype for reference

---

### [v0.1.0] — Initial prototype (`rag.py`)

**Files changed:** `rag.py` (new), `documents/kubernetes_architecture.md` (new)

- **Added** single-file RAG pipeline in `rag.py` covering: document loading,
  sentence splitting, local HuggingFace embedding, ChromaDB persistence, and retrieval
- **Added** `documents/kubernetes_architecture.md` as the initial knowledge base
  covering Kubernetes control plane, worker nodes, and troubleshooting
- **Hardcoded** question: *"Why can a Kubernetes pod remain in Pending state?"*
- **Config values:** `CHUNK_SIZE=200`, `CHUNK_OVERLAP=30`, `similarity_top_k=3`,
  model `sentence-transformers/all-MiniLM-L6-v2`
