
from pathlib import Path

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    DOCUMENT_DIR,
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# DOCUMENT_DIR = Path("documents")
# DB_DIR = Path("chroma_db")
# COLLECTION_NAME = "kubernetes_docs"


def main():
    # 1. Load source documents
    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENT_DIR),
        required_exts=[".md"],
    ).load_data()

    print(f"Loaded documents: {len(documents)}")

    if not documents:
        raise RuntimeError("No Markdown documents found.")

    # 2. Split documents into chunks
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    print(f"Created chunks: {len(nodes)}")

    for i, node in enumerate(nodes, start=1):
        print(f"\n--- Chunk {i} ---")
        print(node.text)

    # 3. Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    # 4. Initialize persistent ChromaDB
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Learning mode: rebuild the collection on each ingestion run
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # 5. Generate embeddings and save vectors
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("\nIngestion completed successfully.")
    print(f"Indexed chunks: {collection.count()}")
    print(f"Database location: {DB_DIR.resolve()}")


if __name__ == "__main__":
    main()
