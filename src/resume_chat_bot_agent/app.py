import os
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from .embedding import EmbeddingClient
from .vector_store import VectorStore, VectorDocument
from .logging_store import QuestionLogger
from .utils import extract_text_from_pdf, extract_text_from_txt


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
STATIC_DIR = ROOT_DIR / "static"

DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Load .env if present for OPENAI_API_KEY and vector settings
load_dotenv()


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


def get_embedding_client() -> EmbeddingClient:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return EmbeddingClient(api_key=api_key)


def get_vector_store() -> VectorStore:
    provider = os.getenv("VECTOR_PROVIDER", "memory").lower()
    if provider == "pinecone":
        index_name = os.getenv("PINECONE_INDEX", "resume-chatbot")
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        if not api_key:
            raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not set")
        return VectorStore.pinecone_backend(index_name=index_name, api_key=api_key, environment=environment)
    return VectorStore.memory_backend(storage_dir=DATA_DIR / "vector_store")


def get_question_logger() -> QuestionLogger:
    return QuestionLogger(db_path=DATA_DIR / "analytics.db")


app = FastAPI(title="Resume Chatbot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index_path))


def _chunk_text(text: str, token_limit: int = 800) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    for paragraph in text.split("\n\n"):
        if sum(len(p) for p in current) + len(paragraph) > token_limit and current:
            chunks.append("\n".join(current))
            current = []
        if paragraph.strip():
            current.append(paragraph.strip())
    if current:
        chunks.append("\n".join(current))
    return chunks


def _preload_knowledge(embedding_client: EmbeddingClient, store: VectorStore) -> int:
    files = list(KNOWLEDGE_DIR.glob("*.pdf")) + list(KNOWLEDGE_DIR.glob("*.txt")) + list(KNOWLEDGE_DIR.glob("*.md"))
    if not files:
        return 0
    # If memory backend, clear existing to avoid duplicates across restarts
    if getattr(store, "_backend", None) == "memory":
        idx = getattr(store, "_index_path", None)
        meta = getattr(store, "_meta_path", None)
        try:
            if idx and Path(idx).exists():
                Path(idx).unlink(missing_ok=True)
            if meta and Path(meta).exists():
                Path(meta).unlink(missing_ok=True)
        except Exception:
            pass
    total_chunks = 0
    for f in files:
        try:
            content = f.read_bytes()
            if f.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(content)
            else:
                text = extract_text_from_txt(content)
            if not text.strip():
                continue
            chunks = _chunk_text(text)
            if not chunks:
                continue
            embeddings = embedding_client.embed_texts(chunks)
            documents = [
                VectorDocument(id=f"{f.name}:{i}", text=chunk, metadata={"source": f.name, "chunk": i})
                for i, chunk in enumerate(chunks)
            ]
            store.upsert(documents=documents, vectors=embeddings)
            total_chunks += len(chunks)
        except Exception:
            continue
    return total_chunks


@app.on_event("startup")
def preload_on_start():
    try:
        emb = get_embedding_client()
        vs = get_vector_store()
        _preload_knowledge(emb, vs)
    except Exception:
        # Don’t block startup; chat may still work even if preload fails
        pass


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...),
                 embedding_client: EmbeddingClient = Depends(get_embedding_client),
                 store: VectorStore = Depends(get_vector_store)):
    file_ext = Path(file.filename).suffix.lower()
    content_bytes = await file.read()
    (KNOWLEDGE_DIR / file.filename).write_bytes(content_bytes)

    if file_ext in [".pdf"]:
        text = extract_text_from_pdf(content_bytes)
    elif file_ext in [".txt", ".md"]:
        text = extract_text_from_txt(content_bytes)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF or TXT.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from file.")

    chunks: List[str] = []
    current: List[str] = []
    token_limit = 800
    for paragraph in text.split("\n\n"):
        if sum(len(p) for p in current) + len(paragraph) > token_limit and current:
            chunks.append("\n".join(current))
            current = []
        current.append(paragraph.strip())
    if current:
        chunks.append("\n".join(current))

    embeddings = embedding_client.embed_texts(chunks)
    documents = [
        VectorDocument(
            id=f"{file.filename}:{i}",
            text=chunk,
            metadata={"source": file.filename, "chunk": i}
        ) for i, chunk in enumerate(chunks)
    ]
    store.upsert(documents=documents, vectors=embeddings)

    return {"chunks": len(chunks), "message": "Ingestion completed"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest,
               embedding_client: EmbeddingClient = Depends(get_embedding_client),
               store: VectorStore = Depends(get_vector_store),
               qlogger: QuestionLogger = Depends(get_question_logger)):
    qlogger.record_question(req.question)
    query_vec = embedding_client.embed_text(req.question)
    results = store.similarity_search(query_vector=query_vec, top_k=req.top_k)

    context_lines: List[str] = []
    sources: List[str] = []
    for item in results:
        context_lines.append(item.document.text)
        src = str(item.document.metadata.get("source"))
        if src not in sources:
            sources.append(src)

    system_prompt = (
        "You are a helpful AI Resume Assistant for a data engineer with Snowflake, DBT, ETL, and AI projects. "
        "Answer clearly with bullet points when helpful. Cite relevant project or resume sections briefly."
    )
    context = "\n\n".join(context_lines[:10])
    answer = embedding_client.generate_answer(system_prompt=system_prompt, context=context, question=req.question)
    return ChatResponse(answer=answer, sources=sources)


@app.get("/api/stats/top-questions")
def top_questions(limit: int = 10, qlogger: QuestionLogger = Depends(get_question_logger)):
    return {"top": qlogger.get_top_questions(limit)}


@app.get("/download/resume")
def download_resume():
    pdfs: List[Path] = list(KNOWLEDGE_DIR.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(status_code=404, detail="No resume PDF found. Upload one via /api/ingest.")
    return FileResponse(path=str(pdfs[0]), filename=pdfs[0].name, media_type="application/pdf")


def main():
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("resume_chat_bot_agent.app:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()


