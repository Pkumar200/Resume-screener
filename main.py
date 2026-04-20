from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import tempfile
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = FastAPI(title="Resume Screener API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR
STATIC_DIR = BASE_DIR / "static"

# Mount frontend static files (optional)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Config ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "resume-screener"
DIMENSION = 1536  # text-embedding-ada-002

_vectorstore: Optional[PineconeVectorStore] = None


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise HTTPException(status_code=500, detail=f"Missing required environment variable: {name}")
    return val


def get_vectorstore() -> PineconeVectorStore:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    openai_key = _require_env("OPENAI_API_KEY")
    pinecone_key = _require_env("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_key)

    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    _vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    return _vectorstore


# ── Pydantic models ──────────────────────────────────────
class ScreenRequest(BaseModel):
    job_description: str
    candidate_name: str = "Candidate"


class ScreenResponse(BaseModel):
    match_score: int  # 0-100
    summary: str
    strengths: List[str]
    gaps: List[str]
    verdict: str  # "Strong Match" | "Good Match" | "Partial Match" | "Weak Match"


# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend not found (missing index.html).")
    return FileResponse(str(index_path))


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), candidate_name: str = "Candidate"):
    """Upload a PDF resume and store chunks in Pinecone."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    vectorstore = get_vectorstore()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = splitter.split_documents(pages)

        # Tag each chunk with the candidate name so we can filter later
        for chunk in chunks:
            chunk.metadata["candidate"] = candidate_name

        vectorstore.add_documents(chunks)
        return {"message": f"Resume for '{candidate_name}' uploaded successfully.", "chunks": len(chunks)}

    finally:
        os.unlink(tmp_path)


@app.post("/screen", response_model=ScreenResponse)
async def screen_resume(req: ScreenRequest):
    """Match a job description against a stored resume using RAG."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6, "filter": {"candidate": req.candidate_name}},
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an expert technical recruiter and resume screener.

Below is relevant content extracted from a candidate's resume:
{context}

Job Description:
{question}

Analyze how well this candidate matches the job description. Respond ONLY with a valid JSON object using this exact structure:
{{
  "match_score": <integer 0-100>,
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "gaps": ["<gap 1>", "<gap 2>"],
  "verdict": "<one of: Strong Match | Good Match | Partial Match | Weak Match>"
}}

Be honest, specific, and cite actual skills from the resume. Do not invent skills that are not present.
""",
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=_require_env("OPENAI_API_KEY"),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
    )

    raw = qa_chain.invoke({"query": req.job_description})
    result_text = raw["result"].strip()

    # Strip markdown code fences if present
    if result_text.startswith("```"):
        result_text = result_text.split("```")[1]
        if result_text.startswith("json"):
            result_text = result_text[4:]
        result_text = result_text.strip()

    import json

    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON. Try again.")

    return ScreenResponse(**data)


@app.delete("/clear-resume/{candidate_name}")
async def clear_resume(candidate_name: str):
    """Delete all vectors for a candidate from Pinecone."""
    vectorstore = get_vectorstore()
    vectorstore._index.delete(filter={"candidate": candidate_name})
    return {"message": f"Vectors for '{candidate_name}' deleted."}


@app.get("/health")
def health():
    return {"status": "ok"}
