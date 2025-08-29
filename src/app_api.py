# app_api.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import time
import logging
import cProfile
import pstats
import io
from functools import wraps

from lib.rag_service import answer_query
from langchain.schema import Document
import lib.streamlit.session_state as ss
import lib.langchain.embeddings as langchain_embeddings
import lib.langchain.opensearch as langchain_opensearch

# Configure logging for timing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
timing_logger = logging.getLogger("timing")


class FileUpload(BaseModel):
    name: str
    content: str


class QueryRequest(BaseModel):
    query: str
    embedding_model: str
    llm_model: str
    use_rag: bool = True
    max_documents: int = 5
    score_threshold: float = 0.0
    use_opensearch: bool = False
    use_qdrant: bool = False
    prompt: str
    files: List[FileUpload] = []
    history: List[Dict[str, str]] = []
    knowledge_base: str = "default"


class KBInspectRequest(BaseModel):
    knowledge_base: str = "default"
    embedding_model: str = "huggingface:sentence-transformers/all-mpnet-base-v2"
    use_opensearch: bool = False


app = FastAPI()

# Timing middleware to instrument request performance
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.time()
    timing_logger.info(f"🚀 REQUEST START: {request.url.path}")
    
    response = await call_next(request)
    
    end_time = time.time()
    duration = end_time - start_time
    timing_logger.info(f"✅ REQUEST END: {request.url.path} - Duration: {duration:.3f}s")
    
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain like ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],  # or ["POST"]
    allow_headers=["*"],
)

# Profiling decorator
def profile_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        # Add detailed timing
        start_time = time.time()
        timing_logger.info(f"📊 PROFILING START: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            timing_logger.info(f"📊 PROFILING END: {func.__name__} - Duration: {duration:.3f}s")
            
            pr.disable()
            
            # Generate profile stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            timing_logger.info(f"📈 PROFILE STATS:\n{s.getvalue()}")
            
            return result
            
        except Exception as e:
            pr.disable()
            timing_logger.error(f"❌ PROFILING ERROR: {func.__name__} - {str(e)}")
            raise
            
    return wrapper

@app.post("/ask")
@profile_request
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
        use_rag=req.use_rag,
        knowledge_base=req.knowledge_base,
        use_qdrant=req.use_qdrant,
    )
    # only return doc metadata, not full text
    doc_info = [
        {"id": d.id, "score": score, "snippet": d.page_content[:200]}
        for d, score in docs
    ]
    return {"answer": answer, "documents": doc_info}


@app.post("/kb/count")
def count_kb_documents(req: KBInspectRequest):
    """Count documents in a knowledge base (works with both OpenSearch and in-memory)"""
    try:
        if req.use_opensearch:
            # Use OpenSearch for counting
            vs = langchain_opensearch.opensearch_doc_vector_store(
                req.embedding_model, req.knowledge_base
            )
            # Get total count by searching with match_all query
            results = vs.similarity_search("", k=1)  # Just to check if any docs exist
            # For more accurate count, we'd need to use the raw OpenSearch client
            count = len(vs.similarity_search("", k=10000))  # Rough estimate
            return {"knowledge_base": req.knowledge_base, "count": count, "storage": "opensearch"}
        else:
            # Use in-memory vector store
            vs = langchain_embeddings.get_in_memory_vector_store(
                req.embedding_model, req.knowledge_base
            )
            if vs is None:
                return {"knowledge_base": req.knowledge_base, "count": 0, "storage": "in-memory", "error": "Knowledge base not found"}
            
            # For in-memory stores, we can get documents directly
            try:
                # Try to get a large number of documents to estimate count
                docs = vs.similarity_search("", k=10000)
                count = len(docs)
            except:
                # Fallback: just indicate if KB exists
                count = "unknown"
            
            return {"knowledge_base": req.knowledge_base, "count": count, "storage": "in-memory"}
    except Exception as e:
        return {"knowledge_base": req.knowledge_base, "count": 0, "error": str(e)}


@app.post("/kb/sample")
def sample_kb_documents(req: KBInspectRequest):
    """Get sample documents from a knowledge base"""
    try:
        if req.use_opensearch:
            vs = langchain_opensearch.opensearch_doc_vector_store(
                req.embedding_model, req.knowledge_base
            )
        else:
            vs = langchain_embeddings.get_in_memory_vector_store(
                req.embedding_model, req.knowledge_base
            )
        
        if vs is None:
            return {"knowledge_base": req.knowledge_base, "documents": [], "error": "Knowledge base not found"}
        
        # Get sample documents
        docs = vs.similarity_search("", k=3)  # Get up to 3 sample docs
        
        sample_docs = []
        for doc in docs:
            sample_docs.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page_number", "N/A"),
                "content_snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "content_length": len(doc.page_content)
            })
        
        storage_type = "opensearch" if req.use_opensearch else "in-memory"
        return {"knowledge_base": req.knowledge_base, "documents": sample_docs, "storage": storage_type}
    
    except Exception as e:
        return {"knowledge_base": req.knowledge_base, "documents": [], "error": str(e)}


@app.post("/kb/docs")
def list_kb_documents(req: KBInspectRequest):
    """List document sources in a knowledge base"""
    try:
        if req.use_opensearch:
            vs = langchain_opensearch.opensearch_doc_vector_store(
                req.embedding_model, req.knowledge_base
            )
        else:
            vs = langchain_embeddings.get_in_memory_vector_store(
                req.embedding_model, req.knowledge_base
            )
        
        if vs is None:
            return {"knowledge_base": req.knowledge_base, "documents": [], "error": "Knowledge base not found"}
        
        # Get all documents (up to a reasonable limit)
        docs = vs.similarity_search("", k=1000)
        
        # Group by source and count chunks
        sources = {}
        doc_list = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
            doc_list.append({
                "source": source,
                "page": doc.metadata.get("page_number", "N/A"),
                "content_length": len(doc.page_content)
            })
        
        storage_type = "opensearch" if req.use_opensearch else "in-memory"
        return {
            "knowledge_base": req.knowledge_base, 
            "documents": doc_list,
            "sources": sources,
            "total_chunks": len(docs),
            "storage": storage_type
        }
    
    except Exception as e:
        return {"knowledge_base": req.knowledge_base, "documents": [], "error": str(e)}


@app.get("/kb/list")
def list_knowledge_bases():
    """List available knowledge bases from both storage types"""
    result = {"opensearch": [], "in_memory": []}
    
    try:
        # Get OpenSearch knowledge bases
        opensearch_kbs = langchain_opensearch.get_available_knowledge_bases()
        result["opensearch"] = opensearch_kbs
    except Exception as e:
        result["opensearch_error"] = str(e)
    
    try:
        # Get in-memory knowledge bases
        in_memory_kbs = langchain_embeddings.get_in_memory_knowledge_bases()
        if not in_memory_kbs:
            in_memory_kbs = ["default"]  # Default is always available
        result["in_memory"] = in_memory_kbs
    except Exception as e:
        result["in_memory_error"] = str(e)
    
    return result
