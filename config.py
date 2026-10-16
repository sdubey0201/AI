
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = BASE_DIR / "documents"
DB_DIR = BASE_DIR / "chroma_db"

# Vector database
COLLECTION_NAME = "kubernetes_docs"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE = 128
CHUNK_OVERLAP = 40

# Retrieval configuration
SIMILARITY_TOP_K = 3

MIN_SIMILARITY_SCORE = 0.35


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

TOKEN_LIMIT = 2000


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


