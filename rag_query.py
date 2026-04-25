"""
RAG Query System — LangChain + ChromaDB
Loads an existing vector store and answers questions using retrieved context.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# ── Configuration ─────────────────────────────────────────────────────────────

VECTOR_DB_DIR   = "./vector_db"             # Must match path used in rag_ingest.py
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"             # Swap for "gpt-4o" for higher quality
TOP_K           = 3                         # Number of chunks to retrieve per query


# ── Prompt template ───────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY the 
context provided below. If the answer is not in the context, say 
"I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


# ── Load vector store ─────────────────────────────────────────────────────────

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )
    print(f"✅ Loaded vector store from: {VECTOR_DB_DIR}")
    return vs


# ── Build RAG chain ───────────────────────────────────────────────────────────

def build_rag_chain(vector_store):
    llm = Ollama(model="llama3.2:latest")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                 # "stuff" = inject all chunks into prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain


# ── Query loop ────────────────────────────────────────────────────────────────

def query_loop(chain):
    print("\n💬 RAG Query System ready. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        result = chain.invoke({"query": question})

        print(f"\n📝 Answer:\n{result['result']}\n")

        print("📚 Sources:")
        seen = set()
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                print(f"   • {src}")
                seen.add(src)
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    vs    = load_vector_store()
    chain = build_rag_chain(vs)
    query_loop(chain)


if __name__ == "__main__":
    main()
