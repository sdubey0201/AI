"""
config.py — Central configuration for the RAG Assistant.

All tuneable values live here. No magic numbers should appear in
ingest.py or query.py — import from this file instead.
"""

from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────

# BASE_DIR: absolute path to the folder containing this file.
# Using __file__ makes all paths work regardless of where you run the script from.
BASE_DIR = Path(__file__).resolve().parent

# DOCUMENT_DIR: folder scanned by ingest.py for .md knowledge base files.
# Drop new Markdown documents here, then re-run ingest.py to index them.
DOCUMENT_DIR = BASE_DIR / "documents"

# DB_DIR: on-disk ChromaDB storage created automatically by ingest.py.
# Do not edit this folder manually — it is managed entirely by ChromaDB.
DB_DIR = BASE_DIR / "chroma_db"

# ── Vector database ───────────────────────────────────────────────────────────

# COLLECTION_NAME: the logical bucket inside ChromaDB that holds all vectors.
# ingest.py writes to this collection; query.py reads from it.
# Changing this name requires a full re-ingestion.
COLLECTION_NAME = "kubernetes_docs"

# ── Embedding model ───────────────────────────────────────────────────────────

# EMBEDDING_MODEL: HuggingFace model used to convert text → 384-dim float vectors.
# Must be the same model in both ingest.py (indexing) and query.py (querying).
# If you change this, delete chroma_db/ and re-run ingest.py — old vectors
# were produced by a different model and will give wrong similarity scores.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking configuration ────────────────────────────────────────────────────

# CHUNK_SIZE: maximum number of tokens per chunk fed to the embedding model.
# Smaller = more precise retrieval but more chunks to store.
# Larger = more context per chunk but less retrieval precision.
CHUNK_SIZE = 128

# CHUNK_OVERLAP: how many tokens the end of one chunk shares with the start
# of the next. Prevents context from being cut mid-sentence at boundaries.
CHUNK_OVERLAP = 40

# ── Retrieval configuration ───────────────────────────────────────────────────

# SIMILARITY_TOP_K: how many candidate chunks ChromaDB returns per query.
# These are the top-k closest vectors by cosine similarity before filtering.
SIMILARITY_TOP_K = 3

# MIN_SIMILARITY_SCORE: quality gate applied after retrieval.
# Any chunk with a cosine similarity score below this value is discarded.
# Range 0.0–1.0. Raise for stricter precision; lower for broader recall.
MIN_SIMILARITY_SCORE = 0.35

# ── Answer generation prompt ──────────────────────────────────────────────────

# RAG_PROMPT_TEMPLATE: the instruction string sent to the LLM for answer generation.
# Placeholders filled at runtime:
#   {context}  — concatenated text of all chunks that passed MIN_SIMILARITY_SCORE
#   {question} — the user's original question
# The "do not invent facts" rule is critical — it prevents the LLM from hallucinating
# information that is not present in the retrieved context.
RAG_PROMPT_TEMPLATE = """
You are a technical knowledge assistant.

Answer the question using only the provided context.

Rules:
- Do not invent facts.
- If the context does not contain the answer, say:
  "I don't know based on the available documents."
- Give a clear, concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

# ── Conversation memory ───────────────────────────────────────────────────────

# TOKEN_LIMIT: maximum number of tokens kept in the ChatMemoryBuffer.
# When the buffer exceeds this limit, the oldest messages are dropped automatically.
# Prevents the conversation history from overflowing the LLM's context window (4096).
TOKEN_LIMIT = 2000

# ── Query rewriting prompt ────────────────────────────────────────────────────

# QUERY_REWRITE_PROMPT_TEMPLATE: sent to the LLM before retrieval when conversation
# history exists. Its job is to rewrite ambiguous follow-up questions into fully
# self-contained standalone questions so ChromaDB can retrieve the right chunks.
# Example: "What about its limits?" → "What are the resource limits of a Kubernetes pod?"
# Placeholders filled at runtime:
#   {history}  — formatted conversation history from ChatMemoryBuffer
#   {question} — the user's current follow-up question
QUERY_REWRITE_PROMPT_TEMPLATE = """
You rewrite follow-up questions into standalone questions.

Use the conversation history to resolve references such as
"it", "its", "they", and "that".

Rules:
- Preserve the user's original intent.
- If the question is already standalone, return it unchanged.
- Do not answer the question.
- Return only the rewritten question.

Conversation history:
{history}

Current question:
{question}

Standalone question:
"""


