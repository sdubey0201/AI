
"""
query.py — Interactive question-answering loop for the RAG Assistant.

Requires ingest.py to have been run at least once to populate chroma_db/.
Requires Ollama to be running locally with llama3.2:latest pulled.

What it does per question:
  1. Reads conversation history from memory
  2. Rewrites follow-up questions into standalone questions (if history exists)
  3. Embeds the rewritten question and searches ChromaDB for similar chunks
  4. Filters out low-confidence results (below MIN_SIMILARITY_SCORE)
  5. Builds a prompt from the retrieved context + conversation history
  6. Sends the prompt to Ollama and prints the synthesised answer
  7. Saves the question + answer to the in-session memory buffer

Usage:
  python query.py
  Type 'exit' or 'quit' to stop.
"""

import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole

from config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MIN_SIMILARITY_SCORE,
    RAG_PROMPT_TEMPLATE,
    TOKEN_LIMIT,
    QUERY_REWRITE_PROMPT_TEMPLATE,
)

# ── Module-level singletons ───────────────────────────────────────────────────
# Initialised once at import time so they are reused across every question
# in the session rather than being recreated on each loop iteration.

# llm: local Ollama instance.
# request_timeout: max seconds to wait for Ollama to respond (long docs can be slow).
# context_window: tells LlamaIndex how many tokens the model can accept at once.
llm = Ollama(
    model="llama3.2:latest",
    request_timeout=120.0,
    context_window=4096,
)

# memory: rolling conversation buffer, capped at TOKEN_LIMIT tokens.
# Oldest messages are automatically dropped when the limit is exceeded.
# Resets when the process exits — history is in-session only.
memory = ChatMemoryBuffer.from_defaults(
    token_limit=TOKEN_LIMIT
)


def main():
    # ── Step 1: Open the existing vector database ─────────────────────────────
    # PersistentClient opens the on-disk ChromaDB written by ingest.py.
    # No data is loaded into memory here — ChromaDB streams it on demand.
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # get_collection raises an exception if the collection does not exist.
    # This means ingest.py has never been run — give a clear error message.
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Index not found. Run 'python ingest.py' first."
        ) from exc

    print(f"Indexed chunks available: {collection.count()}")

    # ── Step 2: Load the embedding model ─────────────────────────────────────
    # Must be the exact same model used in ingest.py.
    # The question is converted to a vector by this model, then compared
    # against the vectors stored by ingest.py using cosine similarity.
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    # ── Step 3: Connect LlamaIndex to the existing ChromaDB collection ────────
    # ChromaVectorStore wraps the collection in the LlamaIndex VectorStore interface.
    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    # from_vector_store loads the index from the existing stored vectors —
    # it does NOT re-embed anything. This is fast and the key benefit of
    # separating ingestion from querying.
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # ── Step 4: Build the retriever ───────────────────────────────────────────
    # as_retriever returns the top-3 most similar chunks for any given query.
    # similarity_top_k=3 is the number of candidates BEFORE the MIN_SIMILARITY_SCORE
    # filter is applied — so up to 3 chunks, possibly fewer after filtering.
    retriever = index.as_retriever(similarity_top_k=3)

    # ── Step 5: Interactive question loop ────────────────────────────────────
    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        # Allow the user to exit cleanly without a KeyboardInterrupt.
        if question.lower() in {"exit", "quit"}:
            break

        # Skip empty input (e.g. accidental Enter press).
        if not question:
            continue

        # ── Step 6: Read conversation history ────────────────────────────────
        # memory.get() returns all ChatMessage objects stored so far this session.
        # This is read BEFORE retrieval so we can use it to rewrite the question.
        chat_history = memory.get()

        # Format history as "role: content" lines for injection into prompts.
        history_text = "\n".join(
            f"{message.role.value}: {message.content}"
            for message in chat_history
        )

        # ── Step 7: Rewrite follow-up questions ──────────────────────────────
        # Start with the original question as the default search query.
        rewritten_question = question

        if chat_history:
            # Only rewrite when there is prior context to resolve against.
            # If history is empty this is the first question — no rewriting needed.
            try:
                rewrite_prompt = (
                    QUERY_REWRITE_PROMPT_TEMPLATE.format(
                        history=history_text,
                        question=question,
                    )
                )

                # Ask the LLM to produce a standalone version of the question.
                rewrite_response = llm.complete(rewrite_prompt)
                candidate_question = rewrite_response.text.strip()

                # Only use the rewritten version if the LLM returned something.
                if candidate_question:
                    rewritten_question = candidate_question

            except Exception as exc:
                # Rewriting is optional — fall back to the original question
                # rather than crashing the whole query loop.
                print(
                    f"\nQuery rewriting failed: {exc}"
                )
                print("Using the original question instead.")

        # Show both versions so the user can see what was sent to ChromaDB.
        print(f"\nOriginal question: {question}")
        print(f"Search question: {rewritten_question}")

        # ── Step 8: Retrieve relevant chunks from ChromaDB ───────────────────
        # The rewritten question is embedded and compared against all stored
        # vectors. Top-3 closest chunks by cosine similarity are returned.
        results = retriever.retrieve(rewritten_question)

        # DEBUG: print raw results before filtering so you can inspect scores
        # and decide whether to tune MIN_SIMILARITY_SCORE, CHUNK_SIZE, or CHUNK_OVERLAP.
        print("\nDebug: Retrieved results before filtering")

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Similarity score: {result.score}")
            print(result.node.get_content())

        # ── Step 9: Filter by similarity threshold ────────────────────────────
        # Discard any result whose score is below MIN_SIMILARITY_SCORE.
        # This prevents the LLM from receiving unrelated context that happens
        # to be the "least bad" match in the vector space.
        relevant_results = [
            result
            for result in results
            if result.score is not None
            and result.score >= MIN_SIMILARITY_SCORE
        ]

        # DEBUG: show how many chunks survived the filter.
        print(f"\nConfigured threshold: {MIN_SIMILARITY_SCORE}")
        print(f"Results before filtering: {len(results)}")
        print(f"Results after filtering: {len(relevant_results)}")

        # If no chunks passed the threshold, the question is outside the
        # knowledge base — tell the user clearly and wait for the next question.
        if not relevant_results:
            print(
                "\nAnswer: No relevant information found "
                "in the knowledge base."
            )
            continue

        # ── Step 10: Build context string ────────────────────────────────────
        # Join all passing chunk texts with a blank line between them.
        # This becomes the {context} placeholder in RAG_PROMPT_TEMPLATE.
        context = "\n\n".join(
            result.node.get_content()
            for result in relevant_results
        )

        # ── Step 11: Build the full answer prompt ────────────────────────────
        # First fill RAG_PROMPT_TEMPLATE with the retrieved context and question.
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,  # use original question, not rewritten
        )

        # Then prepend the conversation history so the LLM can refer to
        # earlier turns when formulating the answer.
        prompt = f"""
Conversation history:
{history_text if history_text else "No previous conversation."}

{prompt}
"""

        # ── Step 12: Generate the answer with Ollama ──────────────────────────
        try:
            # llm.complete() sends the prompt to Ollama's local HTTP server
            # (localhost:11434) and waits for the full response.
            response = llm.complete(prompt)
            answer = response.text.strip()

            print("\nAnswer:")
            print(answer)

            # ── Step 13: Save interaction to memory ──────────────────────────
            # Store both sides of the conversation so future questions can
            # reference this turn via the query rewriting step.
            # USER message = what the user actually typed (not the rewritten form).
            # ASSISTANT message = what the LLM answered.
            memory.put(
                ChatMessage(
                    role=MessageRole.USER,
                    content=question,
                )
            )
            memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=answer,
                )
            )

            # ── Step 14: Show source attribution ─────────────────────────────
            # Display which document file(s) the answer was drawn from,
            # along with the similarity score for each source chunk.
            # file_name is set by SimpleDirectoryReader as metadata on each node.
            print("\nSources:")

            for rank, result in enumerate(
                relevant_results, start=1
            ):
                source = result.node.metadata.get(
                    "file_name", "Unknown source"
                )
                print(
                    f"{rank}. {source} "
                    f"(score: {result.score:.3f})"
                )

        except Exception as exc:
            # Ollama errors (model not found, server not running, timeout)
            # are caught here so the loop continues rather than crashing.
            print(f"\nAnswer generation failed: {exc}")
            print(
                "Check that Ollama is running and "
                "llama3.2:latest is installed."
            )


if __name__ == "__main__":
    main()
