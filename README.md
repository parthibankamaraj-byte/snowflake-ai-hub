# Resume Chatbot (FastAPI + Vector DB + Minimal UI)

This project now includes a FastAPI backend, a minimal static web UI, and a vector store for chatting with your resume and projects.

## Quick Start

1) Create a Python 3.11 virtualenv to avoid native build issues on Windows:

```bash
uv venv --python 3.11 .venv --clear
.venv\\Scripts\\Activate.ps1
```

2) Set environment variables (PowerShell example):

```powershell
$env:OPENAI_API_KEY = "sk-..."
# Optional Pinecone managed vector DB
# $env:VECTOR_PROVIDER = "pinecone"
# $env:PINECONE_API_KEY = "pcn-..."
# $env:PINECONE_INDEX = "resume-chatbot"
```

3) Install deps and run locally:

```bash
uv sync
uv run python -m resume_chat_bot_agent.app
```

Open `http://localhost:8000`.

## Features
- Upload PDF/TXT resume or project notes → chunks + embed
- Chat endpoint uses RAG over your uploaded content
- Top questions are logged to SQLite and viewable in UI
- Download your uploaded resume via `/download/resume`

## Deploy
- HuggingFace Spaces: Use Python FastAPI template, set `app = resume_chat_bot_agent.app.app`
- Render/Fly: Deploy FastAPI service; serve `static/` as root
- Vercel: Deploy `static/` as frontend and point to a hosted FastAPI API

