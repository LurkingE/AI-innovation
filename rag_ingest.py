"""
RAG Ingestion System — LangChain + ChromaDB
Reads all .txt files from a folder, chunks them, embeds them, and saves to a vector store.
Supports both English and Spanish documents.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ── Configuration ─────────────────────────────────────────────────────────────

DOCS_FOLDER   = "C:/Users/opoku/OneDrive/Desktop/tenant_rights"  # Folder with .txt files (EN + ES)
VECTOR_DB_DIR = "./vector_db"   # Where the vector store will be saved
CHUNK_SIZE    = 500             # Characters per chunk
CHUNK_OVERLAP = 50              # Overlap between chunks to preserve context

# all-MiniLM-L6-v2 is multilingual-capable for retrieval;
# swap for "paraphrase-multilingual-MiniLM-L12-v2" for stronger Spanish retrieval.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Step 1: Load all .txt files from the folder ───────────────────────────────

def load_documents(folder: str):
    print(f"\n📂 Loading documents from: {folder}")

    if not Path(folder).exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Try UTF-8 first; fall back to latin-1 to handle Spanish special characters
    # (á, é, í, ó, ú, ñ, ü, ¡, ¿) that may be saved in legacy encodings.
    docs = []
    for encoding in ("utf-8", "latin-1"):
        try:
            loader = DirectoryLoader(
                folder,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": encoding},
                show_progress=True,
            )
            docs = loader.load()
            print(f"✅ Loaded {len(docs)} document(s) using {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"⚠️  {encoding} failed, retrying with latin-1...")

    if not docs:
        raise RuntimeError("Could not load documents — check file encodings.")

    return docs


# ── Step 2: Split documents into chunks ───────────────────────────────────────

def chunk_documents(docs):
    print(f"\n✂️  Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Spanish uses period+space like English; paragraph breaks work for both
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunk(s)")
    return chunks


# ── Step 3: Embed and store in ChromaDB ───────────────────────────────────────

def build_vector_store(chunks, persist_dir: str):
    print(f"\n🔢 Embedding chunks with '{EMBEDDING_MODEL}'...")
    print("   ℹ️  For stronger Spanish retrieval, consider swapping to:")
    print("      paraphrase-multilingual-MiniLM-L12-v2")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"✅ Vector store saved to: {persist_dir}")
    return vector_store


# ── Step 4: Test a sample query in both languages ─────────────────────────────

def test_retrieval(vector_store):
    test_queries = [
        ("en", "What are a tenant's rights?"),
        ("es", "¿Cuáles son los derechos del inquilino?"),
    ]

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    for lang, query in test_queries:
        print(f"\n🔍 [{lang.upper()}] Test query: '{query}'")
        results = retriever.invoke(query)
        print(f"   Top {len(results)} result(s):\n")
        for i, doc in enumerate(results, 1):
            source  = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"   [{i}] Source: {source}")
            print(f"       Snippet: {snippet}...\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    docs   = load_documents(DOCS_FOLDER)
    chunks = chunk_documents(docs)
    vs     = build_vector_store(chunks, VECTOR_DB_DIR)
    test_retrieval(vs)

    print("\n🎉 Ingestion complete! Your vector store is ready for bilingual RAG queries.")


if __name__ == "__main__":
    main()
