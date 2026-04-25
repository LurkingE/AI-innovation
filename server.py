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

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are TANY, a helpful tenant rights assistant. Answer the question using ONLY the 
context provided below. If the answer is not in the context, say 
"I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


# ── Load RAG chain ────────────────────────────────────────────────────────────

def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )
    llm = Ollama(model=OLLAMA_MODEL)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain

print("Loading TANY...")
chain = load_chain()
print("TANY is ready!")


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


@app.post("/ask")
def ask(request: QueryRequest):
    result = chain.invoke({"query": request.question})
    answer = result["result"]
    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    })
    return {"answer": answer, "sources": sources}


@app.get("/")
def index():
    return FileResponse("index.html")
