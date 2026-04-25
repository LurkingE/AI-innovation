"""
TANY — FastAPI Backend
Run with: uvicorn server:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ── Configuration ─────────────────────────────────────────────────────────────

VECTOR_DB_DIR   = "./vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL    = "llama3.2:latest"
TOP_K           = 3

# ── Bilingual RAG Prompts ─────────────────────────────────────────────────────

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


# ── Language Detection ────────────────────────────────────────────────────────

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
    spanish_hits = sum(1 for w in words if w.strip("¿?¡!.,;:") in SPANISH_INDICATORS)
    # If 2+ Spanish function words found, treat as Spanish
    return "es" if spanish_hits >= 2 else "en"


# ── Load RAG chains ───────────────────────────────────────────────────────────

def load_chains():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )
    llm = Ollama(model=OLLAMA_MODEL)

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

print("Loading TANY...")
chain_en, chain_es = load_chains()
print("TANY is ready! (English + Spanish)")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    language: str = "auto"   # "en", "es", or "auto" (auto-detect from question text)


@app.post("/ask")
def ask(request: QueryRequest):
    # Determine language
    if request.language == "auto":
        lang = detect_language(request.question)
    else:
        lang = request.language if request.language in ("en", "es") else "en"

    chain = chain_es if lang == "es" else chain_en
    result = chain.invoke({"query": request.question})

    answer = result["result"]
    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    })
    return {"answer": answer, "sources": sources, "language": lang}


@app.get("/")
def index():
    return FileResponse("index.html")
