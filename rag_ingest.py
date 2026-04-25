"""
RAG Ingestion System — LangChain + ChromaDB
Reads all .txt files from a folder, chunks them, embeds them, and saves to a vector store.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ── Configuration ─────────────────────────────────────────────────────────────

DOCS_FOLDER   = "C:/Users/opoku/OneDrive/Desktop/tenant_rights"       # Folder containing your .txt files
VECTOR_DB_DIR = "./vector_db"       # Where the vector store will be saved
CHUNK_SIZE    = 500                 # Characters per chunk
CHUNK_OVERLAP = 50                  # Overlap between chunks to preserve context
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model


# ── Step 1: Load all .txt files from the folder ───────────────────────────────

def load_documents(folder: str):
    print(f"\n📂 Loading documents from: {folder}")

    if not Path(folder).exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    loader = DirectoryLoader(
        folder,
        glob="**/*.txt",            # Recursively find all .txt files
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    docs = loader.load()
    print(f"✅ Loaded {len(docs)} document(s)")
    return docs


# ── Step 2: Split documents into chunks ───────────────────────────────────────

def chunk_documents(docs):
    print(f"\n✂️  Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Tries to split on paragraphs first
    )

    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunk(s)")
    return chunks


# ── Step 3: Embed and store in ChromaDB ───────────────────────────────────────

def build_vector_store(chunks, persist_dir: str):
    print(f"\n🔢 Embedding chunks with '{EMBEDDING_MODEL}'...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"✅ Vector store saved to: {persist_dir}")
    return vector_store


# ── Step 4: Test a sample query ───────────────────────────────────────────────

def test_retrieval(vector_store, query: str = "What is this document about?"):
    print(f"\n🔍 Test query: '{query}'")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)

    print(f"   Top {len(results)} result(s):\n")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"   [{i}] Source: {source}")
        print(f"       Snippet: {snippet}...\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Export it with:\n"
            "  export OPENAI_API_KEY=sk-..."
        )

    docs   = load_documents(DOCS_FOLDER)
    chunks = chunk_documents(docs)
    vs     = build_vector_store(chunks, VECTOR_DB_DIR)
    test_retrieval(vs)

    print("\n🎉 Ingestion complete! Your vector store is ready for RAG queries.")


if __name__ == "__main__":
    main()
