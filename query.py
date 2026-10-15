
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

from config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MIN_SIMILARITY_SCORE,
    RAG_PROMPT_TEMPLATE,
)

llm = Ollama(
    model="llama3.2:latest",
    request_timeout=120.0,
    context_window=4096,
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

        # 6. Retrieve relevant chunks
        results = retriever.retrieve(question)
        print("\nDebug: Retrieved results before filtering")

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Similarity score: {result.score}")
            print(result.node.get_content())

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

        # 7. Build context
        context = "\n\n".join(
            result.node.get_content()
            for result in relevant_results
        )

        # 8. Build prompt using config.py
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # 9. Generate answer with local Ollama model
        try:
            response = llm.complete(prompt)

            print("\nAnswer:")
            print(response.text.strip())

            # 10. Display source information
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
