# app_api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from lib.rag_service import answer_query
from langchain.schema import Document


class FileUpload(BaseModel):
    name: str
    content: str


class QueryRequest(BaseModel):
    query: str
    embedding_model: str
    llm_model: str
    max_documents: int = 5
    score_threshold: float = 0.0
    use_opensearch: bool = False
    prompt: str
    files: List[FileUpload] = []
    history: List[Dict[str, str]] = []


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain like ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],  # or ["POST"]
    allow_headers=["*"],
)

@app.post("/ask")
def ask(req: QueryRequest):
    user_files = [(f.name, f.content) for f in req.files]
    answer, docs = answer_query(
        query=req.query,
        embedding_model=req.embedding_model,
        llm_model=req.llm_model,
        k=req.max_documents,
        score_threshold=req.score_threshold,
        use_opensearch=req.use_opensearch,
        prompt_template=req.prompt,
        user_files=user_files,
        history=req.history,
    )
    # only return doc metadata, not full text
    doc_info = [
        {"id": d.id, "score": score, "snippet": d.page_content[:200]}
        for d, score in docs
    ]
    return {"answer": answer, "documents": doc_info}
