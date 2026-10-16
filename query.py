
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

llm = Ollama(
    model="llama3.2:latest",
    request_timeout=120.0,
    context_window=4096,
)

memory = ChatMemoryBuffer.from_defaults(
    token_limit=TOKEN_LIMIT
)


def main():
    # 1. Open the existing vector database
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Index not found. Run 'python ingest.py' first."
        ) from exc

    print(f"Indexed chunks available: {collection.count()}")

    # 2. Load the same embedding model used during ingestion
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    # 3. Connect LlamaIndex to the existing collection
    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # 4. Create the retriever
    retriever = index.as_retriever(similarity_top_k=3)

    # 5. Accept questions repeatedly
    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        # 6. Read conversation history BEFORE retrieval
        chat_history = memory.get()

        history_text = "\n".join(
            f"{message.role.value}: {message.content}"
            for message in chat_history
        )

        # 7. Rewrite follow-up questions before retrieval
        rewritten_question = question

        if chat_history:
            try:
                rewrite_prompt = (
                    QUERY_REWRITE_PROMPT_TEMPLATE.format(
                        history=history_text,
                        question=question,
                    )
                )

                rewrite_response = llm.complete(rewrite_prompt)
                candidate_question = rewrite_response.text.strip()

                if candidate_question:
                    rewritten_question = candidate_question

            except Exception as exc:
                print(
                    f"\nQuery rewriting failed: {exc}"
                )
                print("Using the original question instead.")

        print(f"\nOriginal question: {question}")
        print(f"Search question: {rewritten_question}")

        # 8. Retrieve using the rewritten question
        results = retriever.retrieve(rewritten_question)

        print("\nDebug: Retrieved results before filtering")

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Similarity score: {result.score}")
            print(result.node.get_content())

        # 9. Filter results by similarity threshold
        relevant_results = [
            result
            for result in results
            if result.score is not None
            and result.score >= MIN_SIMILARITY_SCORE
        ]

        print(f"\nConfigured threshold: {MIN_SIMILARITY_SCORE}")
        print(f"Results before filtering: {len(results)}")
        print(f"Results after filtering: {len(relevant_results)}")

        if not relevant_results:
            print(
                "\nAnswer: No relevant information found "
                "in the knowledge base."
            )
            continue

        # 10. Build context from relevant documents
        context = "\n\n".join(
            result.node.get_content()
            for result in relevant_results
        )

        # 11. Build the answer prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        prompt = f"""
Conversation history:
{history_text if history_text else "No previous conversation."}

{prompt}
"""

        # 12. Generate answer using local Ollama
        try:
            response = llm.complete(prompt)
            answer = response.text.strip()

            print("\nAnswer:")
            print(answer)

            # 13. Save successful interaction to memory
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

            # 14. Display source information
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
            print(f"\nAnswer generation failed: {exc}")
            print(
                "Check that Ollama is running and "
                "llama3.2:latest is installed."
            )


if __name__ == "__main__":
    main()
