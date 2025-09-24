# lib/query_logger.py
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import httpx
except ImportError:
    httpx = None

try:
    from opensearchpy import OpenSearch
except ImportError:
    try:
        from opensearch_py import OpenSearch
    except ImportError:
        OpenSearch = None
from langchain.schema import Document

# Configure logging
logger = logging.getLogger(__name__)


class LoggingStatus(Enum):
    PENDING = "pending"
    EVALUATING = "evaluating" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueryMetadata:
    embedding_model: str
    llm_model: str
    knowledge_base: str
    score_threshold: float
    max_documents: int
    use_opensearch: bool
    use_qdrant: bool
    prompt_template: str


@dataclass
class RAGResults:
    documents_count: int
    documents: List[Dict[str, Any]]
    retrieval_time: float


@dataclass
class LLMResponse:
    final_answer: str
    response_time: float
    token_count: int


@dataclass
class RagasEvaluation:
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    evaluation_time: Optional[float] = None
    evaluated_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    total_time: float
    embedding_time: float


@dataclass
class QueryLogEntry:
    id: str
    timestamp: str
    user_query: str
    metadata: QueryMetadata
    rag_results: RAGResults
    llm_response: LLMResponse
    performance_metrics: PerformanceMetrics
    status: LoggingStatus = LoggingStatus.PENDING
    ragas_evaluation: Optional[RagasEvaluation] = None

    @classmethod
    def create_from_api_data(
        cls,
        query: str,
        embedding_model: str,
        llm_model: str,
        knowledge_base: str,
        score_threshold: float,
        max_documents: int,
        use_opensearch: bool,
        use_qdrant: bool,
        prompt_template: str,
        rag_docs: List[Tuple[Document, float]],
        final_answer: str,
        retrieval_time: float,
        response_time: float,
        total_time: float,
        embedding_time: float = 0.0,
        token_count: int = 0,
    ) -> "QueryLogEntry":
        """Create a QueryLogEntry from API data"""
        
        # Convert rag_docs to serializable format
        doc_info = []
        for doc, score in rag_docs:
            doc_info.append({
                "id": getattr(doc, 'id', str(uuid.uuid4())),
                "score": float(score),
                "snippet": doc.page_content[:200] if doc.page_content else "",
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page_number", "N/A")
            })

        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            user_query=query,
            metadata=QueryMetadata(
                embedding_model=embedding_model,
                llm_model=llm_model,
                knowledge_base=knowledge_base,
                score_threshold=score_threshold,
                max_documents=max_documents,
                use_opensearch=use_opensearch,
                use_qdrant=use_qdrant,
                prompt_template=prompt_template
            ),
            rag_results=RAGResults(
                documents_count=len(rag_docs),
                documents=doc_info,
                retrieval_time=retrieval_time
            ),
            llm_response=LLMResponse(
                final_answer=final_answer,
                response_time=response_time,
                token_count=token_count
            ),
            performance_metrics=PerformanceMetrics(
                total_time=total_time,
                embedding_time=embedding_time
            )
        )

    def to_opensearch_doc(self) -> Dict[str, Any]:
        """Convert to OpenSearch document format"""
        doc = asdict(self)
        # Convert enum to string
        doc['status'] = self.status.value
        return doc


class RagasEvaluatorClient:
    def __init__(self, ragas_base_url: str = "http://localhost:8000"):
        self.base_url = ragas_base_url
        if httpx:
            self.client = httpx.AsyncClient(timeout=60.0)
        else:
            self.client = None
            logger.warning("httpx not available - RAGAS evaluation will be disabled")
        
    async def evaluate_query(self, log_entry: QueryLogEntry) -> RagasEvaluation:
        """Evaluate a query using the ragas evaluator service"""
        if not self.client:
            return RagasEvaluation(error="httpx not available")
            
        try:
            start_time = time.time()
            
            # Prepare ragas evaluation request
            contexts = [doc["snippet"] for doc in log_entry.rag_results.documents]
            
            payload = {
                "question": log_entry.user_query,
                "answer": log_entry.llm_response.final_answer,
                "contexts": contexts,
                "metrics": ["faithfulness", "answer_relevancy"]
            }
            
            logger.info(f"📊 RAGAS EVAL START: query_id={log_entry.id}")
            
            response = await self.client.post(
                f"{self.base_url}/evaluate/single",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                error_msg = f"Ragas API returned {response.status_code}: {response.text}"
                logger.error(f"❌ RAGAS EVAL FAILED: {error_msg}")
                return RagasEvaluation(error=error_msg)
            
            result = response.json()
            evaluation_time = time.time() - start_time
            
            logger.info(f"✅ RAGAS EVAL SUCCESS: query_id={log_entry.id}, time={evaluation_time:.3f}s")
            
            # Extract metrics from the nested response structure
            eval_results = result.get("evaluation_results", {})
            summary = eval_results.get("summary", {})
            
            return RagasEvaluation(
                faithfulness=summary.get("faithfulness", {}).get("mean"),
                answer_relevancy=summary.get("answer_relevancy", {}).get("mean"),
                context_precision=summary.get("context_precision", {}).get("mean"),
                context_recall=summary.get("context_recall", {}).get("mean"),
                evaluation_time=evaluation_time,
                evaluated_at=datetime.utcnow().isoformat() + "Z"
            )
            
        except Exception as e:
            logger.error(f"❌ RAGAS EVAL ERROR: query_id={log_entry.id}, error={str(e)}")
            return RagasEvaluation(error=str(e))

    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()


class QueryLoggerService:
    def __init__(
        self,
        opensearch_host: str = "http://localhost:9200",
        ragas_base_url: str = "http://localhost:8000",
        batch_size: int = 10,
        batch_interval: float = 5.0,
        max_queue_size: int = 1000
    ):
        self.opensearch_host = opensearch_host
        self.ragas_base_url = ragas_base_url
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.max_queue_size = max_queue_size
        
        # Initialize clients
        if OpenSearch:
            self.opensearch_client = OpenSearch(
                hosts=[opensearch_host],
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
                max_retries=3,
                retry_on_timeout=True
            )
        else:
            self.opensearch_client = None
            logger.warning("OpenSearch not available - logging will be disabled")
        self.ragas_client = RagasEvaluatorClient(ragas_base_url)
        
        # Queue for batch processing
        self.log_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_task: Optional[asyncio.Task] = None
        self.evaluation_tasks: List[asyncio.Task] = []
        
        # Track service state
        self.is_running = False
        
        logger.info(f"🔧 QueryLoggerService initialized: batch_size={batch_size}, interval={batch_interval}s")

    def get_index_name(self) -> str:
        """Generate monthly rotating index name"""
        now = datetime.utcnow()
        return f"ouragboros_query_logs_{now.strftime('%Y_%m')}"

    async def ensure_index_exists(self):
        """Ensure the OpenSearch index exists with proper mapping"""
        if not self.opensearch_client:
            return
            
        index_name = self.get_index_name()
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        exists = await loop.run_in_executor(None, self.opensearch_client.indices.exists, index_name)
        
        if exists:
            return
            
        # Define index mapping for better search performance
        mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "user_query": {"type": "text", "analyzer": "standard"},
                    "metadata": {
                        "properties": {
                            "embedding_model": {"type": "keyword"},
                            "llm_model": {"type": "keyword"},
                            "knowledge_base": {"type": "keyword"},
                            "score_threshold": {"type": "float"},
                            "max_documents": {"type": "integer"}
                        }
                    },
                    "rag_results": {
                        "properties": {
                            "documents_count": {"type": "integer"},
                            "retrieval_time": {"type": "float"}
                        }
                    },
                    "llm_response": {
                        "properties": {
                            "response_time": {"type": "float"},
                            "token_count": {"type": "integer"}
                        }
                    },
                    "ragas_evaluation": {
                        "properties": {
                            "faithfulness": {"type": "float"},
                            "answer_relevancy": {"type": "float"},
                            "context_precision": {"type": "float"},
                            "context_recall": {"type": "float"},
                            "evaluation_time": {"type": "float"}
                        }
                    },
                    "status": {"type": "keyword"}
                }
            }
        }
        
        await loop.run_in_executor(None, self.opensearch_client.indices.create, index_name, mapping)
        logger.info(f"✅ Created OpenSearch index: {index_name}")

    async def start(self):
        """Start the logging service"""
        if self.is_running:
            return
            
        self.is_running = True
        await self.ensure_index_exists()
        
        # Start batch processing task
        self.batch_task = asyncio.create_task(self._batch_processor())
        logger.info("🚀 QueryLoggerService started")

    async def stop(self):
        """Stop the logging service gracefully"""
        self.is_running = False
        
        # Cancel batch processing
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
                
        # Cancel evaluation tasks
        for task in self.evaluation_tasks:
            if not task.done():
                task.cancel()
                
        # Wait for remaining items in queue
        await self._flush_queue()
        
        # Close clients
        if hasattr(self.opensearch_client, 'close'):
            self.opensearch_client.close()
        await self.ragas_client.close()
        
        logger.info("🛑 QueryLoggerService stopped")

    async def log_query(self, log_entry: QueryLogEntry):
        """Add a query log entry to the processing queue (non-blocking)"""
        try:
            # Try to add to queue without blocking
            self.log_queue.put_nowait(log_entry)
            logger.debug(f"📝 Queued log entry: {log_entry.id}")
        except asyncio.QueueFull:
            logger.warning(f"⚠️ Log queue full, dropping entry: {log_entry.id}")

    async def _batch_processor(self):
        """Background task that processes log entries in batches"""
        batch = []
        last_flush_time = time.time()
        logger.info(f"🔄 Batch processor started: batch_size={self.batch_size}, interval={self.batch_interval}s")
        
        while self.is_running:
            try:
                # Wait for items or timeout
                try:
                    log_entry = await asyncio.wait_for(
                        self.log_queue.get(), 
                        timeout=1.0
                    )
                    batch.append(log_entry)
                    logger.info(f"📥 Added to batch: {log_entry.id} (batch size: {len(batch)})")
                except asyncio.TimeoutError:
                    # Timeout occurred, check if we should flush
                    pass
                
                current_time = time.time()
                time_since_last_flush = current_time - last_flush_time
                should_flush = (
                    len(batch) >= self.batch_size or 
                    (batch and time_since_last_flush >= self.batch_interval)
                )
                
                if batch:
                    logger.debug(f"🔍 Batch status: size={len(batch)}, time_since_flush={time_since_last_flush:.1f}s, should_flush={should_flush}")
                
                if should_flush and batch:
                    logger.info(f"🚀 Flushing batch of {len(batch)} entries")
                    await self._flush_batch(batch)
                    batch.clear()
                    last_flush_time = current_time
                    
            except Exception as e:
                logger.error(f"❌ Batch processor error: {e}")
                await asyncio.sleep(1)
        
        # Flush remaining items when stopping
        if batch:
            logger.info(f"🔄 Final flush: {len(batch)} entries")
            await self._flush_batch(batch)

    async def _flush_batch(self, batch: List[QueryLogEntry]):
        """Flush a batch of log entries to OpenSearch"""
        if not batch or not self.opensearch_client:
            return
            
        try:
            index_name = self.get_index_name()
            await self.ensure_index_exists()
            
            # Prepare bulk operations
            bulk_ops = []
            for log_entry in batch:
                bulk_ops.extend([
                    {"index": {"_index": index_name, "_id": log_entry.id}},
                    log_entry.to_opensearch_doc()
                ])
            
            # Execute bulk insert
            start_time = time.time()
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.opensearch_client.bulk, bulk_ops)
            flush_time = time.time() - start_time
            
            logger.info(f"💾 BATCH FLUSH: {len(batch)} entries in {flush_time:.3f}s")
            
            # Start evaluation tasks for each entry (non-blocking)
            for log_entry in batch:
                eval_task = asyncio.create_task(self._evaluate_and_update(log_entry))
                self.evaluation_tasks.append(eval_task)
                
            # Clean up completed evaluation tasks
            self.evaluation_tasks = [t for t in self.evaluation_tasks if not t.done()]
            
        except Exception as e:
            logger.error(f"❌ Batch flush failed: {e}")

    async def _evaluate_and_update(self, log_entry: QueryLogEntry):
        """Evaluate a log entry with ragas and update the document"""
        try:
            # Perform ragas evaluation
            evaluation = await self.ragas_client.evaluate_query(log_entry)
            
            # Update the log entry
            log_entry.ragas_evaluation = evaluation
            log_entry.status = LoggingStatus.COMPLETED if evaluation.error is None else LoggingStatus.FAILED
            
            # Update the document in OpenSearch
            index_name = self.get_index_name()
            loop = asyncio.get_event_loop()
            update_body = {
                "doc": {
                    "ragas_evaluation": asdict(evaluation),
                    "status": log_entry.status.value
                }
            }
            await loop.run_in_executor(
                None, 
                lambda: self.opensearch_client.update(
                    index=index_name,
                    id=log_entry.id,
                    body=update_body
                )
            )
            
            logger.info(f"📊 EVAL UPDATED: {log_entry.id}, status={log_entry.status.value}")
            
        except Exception as e:
            logger.error(f"❌ Evaluation update failed for {log_entry.id}: {e}")
            
            # Mark as failed
            try:
                index_name = self.get_index_name()
                loop = asyncio.get_event_loop()
                update_body = {
                    "doc": {
                        "status": LoggingStatus.FAILED.value,
                        "ragas_evaluation": {"error": str(e)}
                    }
                }
                await loop.run_in_executor(
                    None,
                    lambda: self.opensearch_client.update(
                        index=index_name,
                        id=log_entry.id,
                        body=update_body
                    )
                )
            except:
                pass  # Don't fail on the failure update

    async def _flush_queue(self):
        """Flush any remaining items in the queue"""
        remaining_items = []
        while not self.log_queue.empty():
            try:
                item = self.log_queue.get_nowait()
                remaining_items.append(item)
            except asyncio.QueueEmpty:
                break
                
        if remaining_items:
            await self._flush_batch(remaining_items)

    async def search_logs(
        self,
        query: str = "*",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        knowledge_base: Optional[str] = None,
        llm_model: Optional[str] = None,
        vector_store: Optional[str] = None,
        embedding_model: Optional[str] = None,
        status: Optional[str] = None,
        min_ragas_score: Optional[float] = None,
        size: int = 100,
        from_: int = 0
    ) -> Dict[str, Any]:
        """Search query logs with filters"""
        if not self.opensearch_client:
            return {"hits": [], "total": 0, "took": 0, "error": "OpenSearch not available"}
            
        try:
            index_name = self.get_index_name()
            
            # Build search query
            must_clauses = []
            
            if query and query != "*":
                must_clauses.append({
                    "multi_match": {
                        "query": query,
                        "fields": ["user_query", "llm_response.final_answer"]
                    }
                })
            
            if knowledge_base:
                must_clauses.append({
                    "term": {"metadata.knowledge_base": knowledge_base}
                })
                
            if llm_model:
                must_clauses.append({
                    "term": {"metadata.llm_model": llm_model}
                })

            # Vector store filtering based on use_opensearch and use_qdrant boolean fields
            if vector_store:
                if vector_store.lower() == "qdrant":
                    must_clauses.append({
                        "term": {"metadata.use_qdrant": True}
                    })
                elif vector_store.lower() == "opensearch":
                    must_clauses.append({
                        "term": {"metadata.use_opensearch": True}
                    })
                elif vector_store.lower() == "inmemory":
                    # In-memory means both use_opensearch and use_qdrant are false
                    must_clauses.append({
                        "bool": {
                            "must": [
                                {"term": {"metadata.use_opensearch": False}},
                                {"term": {"metadata.use_qdrant": False}}
                            ]
                        }
                    })

            if embedding_model:
                must_clauses.append({
                    "term": {"metadata.embedding_model": embedding_model}
                })

            if status:
                must_clauses.append({
                    "term": {"status": status}
                })

            if min_ragas_score is not None:
                # Filter for records with RAGAS scores above the minimum
                must_clauses.append({
                    "bool": {
                        "should": [
                            {"range": {"ragas_evaluation.faithfulness": {"gte": min_ragas_score}}},
                            {"range": {"ragas_evaluation.answer_relevancy": {"gte": min_ragas_score}}},
                            {"range": {"ragas_evaluation.context_precision": {"gte": min_ragas_score}}},
                            {"range": {"ragas_evaluation.context_recall": {"gte": min_ragas_score}}}
                        ],
                        "minimum_should_match": 1
                    }
                })

            if from_date or to_date:
                date_filter = {}
                if from_date:
                    date_filter["gte"] = from_date
                if to_date:
                    date_filter["lte"] = to_date
                must_clauses.append({
                    "range": {"timestamp": date_filter}
                })
            
            if must_clauses:
                search_body = {
                    "query": {
                        "bool": {"must": must_clauses}
                    },
                    "sort": [{"timestamp": {"order": "desc"}}],
                    "size": size,
                    "from": from_
                }
            else:
                search_body = {
                    "query": {"match_all": {}},
                    "sort": [{"timestamp": {"order": "desc"}}],
                    "size": size,
                    "from": from_
                }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.opensearch_client.search(
                    index=index_name,
                    body=search_body
                )
            )
            
            return {
                "hits": response["hits"]["hits"],
                "total": response["hits"]["total"]["value"],
                "took": response["took"]
            }
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return {"hits": [], "total": 0, "took": 0, "error": str(e)}


# Global instance
_logger_service: Optional[QueryLoggerService] = None


async def get_query_logger() -> QueryLoggerService:
    """Get or create the global query logger instance"""
    global _logger_service
    
    if _logger_service is None:
        # Initialize with environment variables or defaults
        import os
        
        opensearch_host = os.getenv("OPENSEARCH_BASE_URL", "http://localhost:9200") 
        ragas_url = os.getenv("RAGAS_BASE_URL", "http://localhost:8002")
        
        _logger_service = QueryLoggerService(
            opensearch_host=opensearch_host,
            ragas_base_url=ragas_url
        )
        await _logger_service.start()
        
    return _logger_service


async def shutdown_query_logger():
    """Shutdown the global query logger"""
    global _logger_service
    
    if _logger_service:
        await _logger_service.stop()
        _logger_service = None