# 📄 Resume Screener — RAG-Powered Candidate Matching

An AI-powered resume screening tool built with **LangChain**, **Pinecone**, **FastAPI**, and **GPT-4o-mini**.  
Upload a PDF resume, paste a job description, and get an instant match analysis with score, strengths, and skill gaps.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple)

---

## ✨ Features

- 📤 **PDF resume upload** — parsed and chunked automatically
- 🔍 **RAG-based matching** — relevant chunks retrieved from Pinecone vector DB
- 🤖 **GPT-4o-mini analysis** — structured match score, summary, strengths & gaps
- 🎯 **Match score** (0–100) with verdict: Strong / Good / Partial / Weak Match
- 🌐 **Clean web UI** — drag & drop, animated score ring, no framework dependencies

---

## 🏗️ Architecture

```
PDF Resume
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter
    │
    ▼
OpenAI Embeddings (text-embedding-ada-002)
    │
    ▼
Pinecone Vector Store (cosine similarity, filtered by candidate)
    │
    ▼
RetrievalQA Chain (top-6 chunks) + GPT-4o-mini
    │
    ▼
Structured JSON → FastAPI → HTML Frontend
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/pkumaraswamy/resume-screener.git
cd resume-screener
```

### 2. Set up the backend

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   OPENAI_API_KEY=sk-...
#   PINECONE_API_KEY=...
```

Get your keys:
- [OpenAI API Key](https://platform.openai.com/api-keys)
- [Pinecone API Key](https://app.pinecone.io/) — free tier is enough

### 4. Run the server

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Windows (PowerShell) quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and set OPENAI_API_KEY + PINECONE_API_KEY
uvicorn main:app --reload
```

---

## 📖 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload-resume` | Upload PDF resume (multipart/form-data) |
| `POST` | `/screen` | Screen candidate against job description |
| `DELETE` | `/clear-resume/{name}` | Remove candidate vectors from Pinecone |
| `GET` | `/health` | Health check |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.10+ |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector DB | Pinecone (Serverless) |
| RAG Framework | LangChain |
| PDF Parsing | PyPDF |
| Frontend | Vanilla HTML/CSS/JS |

---

## 💡 How It Works

1. **Ingestion** — The resume PDF is loaded, split into 500-token chunks with 80-token overlap, and embedded using OpenAI's embedding model.
2. **Storage** — Chunks are stored in Pinecone with metadata filtering by candidate name.
3. **Retrieval** — When a job description is submitted, the top-6 most relevant resume chunks are retrieved via cosine similarity search.
4. **Generation** — GPT-4o-mini receives the retrieved chunks + job description and returns a structured JSON analysis.

---

## 📸 Screenshot

> Upload a resume → paste a JD → get instant AI analysis with score, verdict, strengths & gaps.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

---

## 📄 License

MIT

---

*Built by [P Kumaraswamy](https://github.com/pkumaraswamy) — Full Stack Developer & AI Engineer*
