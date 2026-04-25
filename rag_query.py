"""
RAG Query System — LangChain + ChromaDB
Loads an existing vector store and answers questions in English or Spanish.
Language is auto-detected from the question, or can be forced with --lang en/es.
"""

import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# ── Configuration ─────────────────────────────────────────────────────────────

VECTOR_DB_DIR   = "./vector_db"         # Must match path used in rag_ingest.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL      = "llama3.2:latest"
TOP_K           = 3


# ── Bilingual prompts ─────────────────────────────────────────────────────────

RAG_PROMPT_EN = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are TANY, a helpful tenant rights assistant. Answer the question using ONLY the 
context provided below. If the answer is not in the context, say 
"I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)

RAG_PROMPT_ES = PromptTemplate(
    input_variables=["context", "question"],
    template="""Eres TANY, un asistente útil sobre derechos de inquilinos. Responde la pregunta usando 
ÚNICAMENTE el contexto proporcionado a continuación. Si la respuesta no está en el contexto, di 
"No tengo suficiente información para responder eso."
Responde SIEMPRE en español.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
)


# ── Language detection ────────────────────────────────────────────────────────

SPANISH_INDICATORS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "es", "son", "está", "están", "ser", "estar",
    "qué", "cómo", "cuándo", "dónde", "por", "para",
    "que", "con", "del", "al", "se", "me", "te", "le",
    "puedo", "puede", "pueden", "tengo", "tiene", "tienen",
    "mi", "tu", "su", "mis", "tus", "sus",
    "si", "no", "sí", "hay", "fue", "han", "sido",
}

def detect_language(text: str) -> str:
    """Return 'es' if text appears to be Spanish, else 'en'."""
    words = text.lower().split()
    hits  = sum(1 for w in words if w.strip("¿?¡!.,;:") in SPANISH_INDICATORS)
    return "es" if hits >= 2 else "en"


# ── Load vector store ─────────────────────────────────────────────────────────

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )
    print(f"✅ Loaded vector store from: {VECTOR_DB_DIR}")
    return vs


# ── Build RAG chains ──────────────────────────────────────────────────────────

def build_rag_chains(vector_store):
    llm = Ollama(model=CHAT_MODEL)

    chain_en = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT_EN},
    )

    chain_es = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT_ES},
    )

    return chain_en, chain_es


# ── Query loop ────────────────────────────────────────────────────────────────

def query_loop(chain_en, chain_es, force_lang: str = "auto"):
    print("\n💬 TANY RAG Query System ready (EN + ES). Type 'quit' to exit.\n")
    if force_lang != "auto":
        print(f"   🔒 Language locked to: {force_lang.upper()}\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye! / ¡Hasta luego!")
            break
        if not question:
            continue

        # Determine language
        lang  = force_lang if force_lang != "auto" else detect_language(question)
        chain = chain_es if lang == "es" else chain_en
        label = "ES 🇲🇽" if lang == "es" else "EN 🇺🇸"

        result = chain.invoke({"query": question})

        print(f"\n[{label}] 📝 Answer:\n{result['result']}\n")

        sources_label = "Fuentes" if lang == "es" else "Sources"
        print(f"📚 {sources_label}:")
        seen = set()
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                print(f"   • {src}")
                seen.add(src)
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Optional CLI flag: python rag_query.py --lang es
    force_lang = "auto"
    if "--lang" in sys.argv:
        idx = sys.argv.index("--lang")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1] in ("en", "es"):
            force_lang = sys.argv[idx + 1]

    vs               = load_vector_store()
    chain_en, chain_es = build_rag_chains(vs)
    query_loop(chain_en, chain_es, force_lang)


if __name__ == "__main__":
    main()
