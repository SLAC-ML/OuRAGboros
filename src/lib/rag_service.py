# lib/rag_service.py
import itertools
from typing import List, Dict, Tuple, AsyncGenerator, Any
import time
import logging
import asyncio

from langchain_community.vectorstores.opensearch_vector_search import (
    SCRIPT_SCORING_SEARCH,
)
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

import lib.config as config
import lib.streamlit.session_state as ss  # for StateKey
import lib.langchain.llm as langchain_llm
import lib.langchain.models as langchain_models
import lib.langchain.opensearch as langchain_opensearch
import lib.langchain.util as langchain_utils

# Configure timing logger
timing_logger = logging.getLogger("timing")


def perform_document_retrieval(
    query: str,
    embedding_model: str,
    k: int,
    score_threshold: float,
    use_opensearch_vectorstore: bool,
    knowledge_base: str = "default",
    use_qdrant: bool = False,
) -> List[Tuple[Document, float]]:
    start_time = time.time()
    timing_logger.info(f"🔍 DOCUMENT RETRIEVAL START: kb={knowledge_base}, qdrant={use_qdrant}, opensearch={use_opensearch_vectorstore}")
    
    vs_start = time.time()
    vs = ss.get_vector_store(use_opensearch_vectorstore, embedding_model, knowledge_base, use_qdrant)
    vs_time = time.time() - vs_start
    timing_logger.info(f"📦 VECTOR STORE GET: {vs_time:.3f}s")
    
    search_start = time.time()
    if use_qdrant:
        # Qdrant handles score threshold internally and uses cosine similarity
        timing_logger.info(f"🎯 QDRANT SEARCH START: query='{query[:50]}...', k={k}")
        results = vs.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=score_threshold,
        )
    elif use_opensearch_vectorstore:
        timing_logger.info(f"🔎 OPENSEARCH SEARCH START: query='{query[:50]}...', k={k}")
        index_start = time.time()
        langchain_opensearch.ensure_opensearch_index(embedding_model, knowledge_base)
        index_time = time.time() - index_start
        timing_logger.info(f"📋 INDEX ENSURE: {index_time:.3f}s")
        
        results = vs.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=score_threshold,
            search_type=SCRIPT_SCORING_SEARCH,
            space_type="cosinesimil",
        )
    else:
        timing_logger.info(f"🧠 INMEMORY SEARCH START: query='{query[:50]}...', k={k}")
        results = [
            (d, s + 1)
            for (d, s) in vs.similarity_search_with_score(query, k=k)
            if s + 1 >= score_threshold
        ]
    
    search_time = time.time() - search_start
    total_time = time.time() - start_time
    timing_logger.info(f"🔍 DOCUMENT RETRIEVAL END: found {len(results)} docs, search={search_time:.3f}s, total={total_time:.3f}s")
    
    return results


def make_system_message(
    prompt: str,
    docs: List[Tuple[Document, float]],
    user_files: List[Tuple[str, str]],
) -> str:
    # Flatten docs + uploaded files into sources
    sources = [(doc.id, doc.page_content) for doc, _ in docs] + user_files
    if not sources:
        return prompt
    blocks = [prompt, "\nSources:"]
    for name, txt in sources:
        blocks.append(f"---{name}---\n{txt}")
    return "\n\n".join(blocks)


def answer_query(
    query: str,
    embedding_model: str,
    llm_model: str,
    k: int,
    score_threshold: float,
    use_opensearch: bool,
    prompt_template: str,
    user_files: List[Tuple[str, str]] = None,
    history: List[Dict[str, str]] = None,
    use_rag: bool = True,
    knowledge_base: str = "default",
    use_qdrant: bool = False,
) -> Tuple[str, List[Tuple[Document, float]]]:
    """
    Returns (full_answer_text, retrieved_docs), streaming is only in UI.
    """
    start_time = time.time()
    timing_logger.info(f"🤖 ANSWER QUERY START: use_rag={use_rag}, model={llm_model}")
    
    # 1. retrieve
    if use_rag:
        retrieval_start = time.time()
        docs = perform_document_retrieval(
            query, embedding_model, k, score_threshold, use_opensearch, knowledge_base, use_qdrant
        )
        retrieval_time = time.time() - retrieval_start
        timing_logger.info(f"📚 RAG RETRIEVAL TOTAL: {retrieval_time:.3f}s")
    else:
        # if not using RAG, return an empty docs list
        timing_logger.info(f"⚡ NO RAG - skipping retrieval")
        docs = []

    # 2. build the base system message
    msg_start = time.time()
    system_msg = make_system_message(prompt_template, docs, user_files or [])
    msg_time = time.time() - msg_start
    timing_logger.info(f"📝 SYSTEM MESSAGE BUILD: {msg_time:.3f}s, msg_len={len(system_msg)}")

    # 3. if there *is* prior chat‐history, stitch it in
    if history:
        # turn [{"role":"user","content":"…"}, …] into text
        hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        system_msg += "\n\nConversation history:\n" + hist_text

    # 4. ensure models are loaded
    model_start = time.time()
    langchain_models.pull_model(embedding_model)
    langchain_models.pull_model(llm_model)
    model_time = time.time() - model_start
    timing_logger.info(f"🔧 MODEL LOADING: {model_time:.3f}s")

    # 5. query
    llm_start = time.time()
    timing_logger.info(f"🧠 LLM QUERY START: model={llm_model}")
    
    # langchain_llm.query_llm yields a generator of tokens
    try:
        tokens, tokens_for_save = itertools.tee(
            langchain_llm.query_llm(llm_model, query, system_msg), 2
        )
        model_source, _ = langchain_utils.parse_model_name(llm_model)
        if model_source in ["openai", "stanford", "google"]:
            answer = "".join(chunk.content for chunk in tokens_for_save)
        else:
            answer = "".join(tokens_for_save)
        
        llm_time = time.time() - llm_start
        timing_logger.info(f"🧠 LLM QUERY SUCCESS (primary): {llm_time:.3f}s, answer_len={len(answer)}")
            
    except Exception as e:
        # instantiate a streaming chat LLM
        timing_logger.info(f"⚠️ LLM QUERY FALLBACK: {str(e)[:100]}...")
        fallback_start = time.time()
        model_source, model_name = langchain_utils.parse_model_name(llm_model)

        if model_source == "openai":
            client_create_time = time.time()
            chat_llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                streaming=True,
            )
            timing_logger.info(f"🏗️ OPENAI CLIENT CREATE: {time.time() - client_create_time:.3f}s")
        elif model_source == "google":
            chat_llm = ChatGoogleGenerativeAI(
                model=model_name,
                # google_api_key=config.google_api_key,
                temperature=0.0,
                streaming=True,
            )
        elif model_source == "stanford":
            # Stanford API uses OpenAI-compatible interface
            client_create_time = time.time()
            chat_llm = ChatOpenAI(
                model=model_name,
                openai_api_key=config.stanford_api_key,
                openai_api_base=config.stanford_base_url,
                temperature=0.0,
                streaming=True,
            )
            timing_logger.info(f"🏗️ CHATopenai CLIENT CREATE: {time.time() - client_create_time:.3f}s")
        else:
            raise RuntimeError(f"Unsupported fallback model source: {model_source}")

        # wrap your system prompt and user query
        msg_wrap_time = time.time()
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=query),
        ]
        timing_logger.info(f"🔧 FALLBACK MSG WRAP: {time.time() - msg_wrap_time:.3f}s")

        # get a token generator and tee it
        stream_start_time = time.time()
        token_stream = chat_llm.stream(messages)
        tokens, tokens_for_save = itertools.tee(token_stream, 2)
        timing_logger.info(f"🌊 FALLBACK STREAM START: {time.time() - stream_start_time:.3f}s")

        # collect the full answer
        collect_start_time = time.time()
        if model_source in ["google", "openai", "stanford"]:
            answer = "".join(chunk.content for chunk in tokens_for_save)
        else:
            answer = "".join(tokens_for_save)
        collect_time = time.time() - collect_start_time
        timing_logger.info(f"📝 FALLBACK COLLECT: {collect_time:.3f}s")
            
        llm_time = time.time() - llm_start
        fallback_time = time.time() - fallback_start
        timing_logger.info(f"🧠 LLM QUERY SUCCESS (fallback): fallback_setup={fallback_time:.3f}s, total_llm={llm_time:.3f}s, answer_len={len(answer)}")

    total_time = time.time() - start_time
    timing_logger.info(f"🤖 ANSWER QUERY END: total={total_time:.3f}s")
    return answer, docs


async def answer_query_stream(
    query: str,
    embedding_model: str,
    llm_model: str,
    k: int,
    score_threshold: float,
    use_opensearch: bool,
    prompt_template: str,
    user_files: List[Tuple[str, str]] = None,
    history: List[Dict[str, str]] = None,
    use_rag: bool = True,
    knowledge_base: str = "default",
    use_qdrant: bool = False,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async streaming version of answer_query that yields tokens as they are generated.
    Yields dictionaries with different types of updates throughout the process.
    """
    start_time = time.time()
    timing_logger.info(f"🌊 ANSWER QUERY STREAM START: use_rag={use_rag}, model={llm_model}")
    
    try:
        # 1. Document retrieval phase
        if use_rag:
            yield {"type": "status", "message": "Retrieving relevant documents..."}
            retrieval_start = time.time()
            
            docs = perform_document_retrieval(
                query, embedding_model, k, score_threshold, use_opensearch, knowledge_base, use_qdrant
            )
            
            retrieval_time = time.time() - retrieval_start
            timing_logger.info(f"📚 STREAM RAG RETRIEVAL: {retrieval_time:.3f}s")
            
            # Send document information
            doc_info = [
                {"id": d.id, "score": score, "snippet": d.page_content[:200]}
                for d, score in docs
            ]
            yield {
                "type": "documents", 
                "data": doc_info,
                "count": len(docs),
                "retrieval_time": retrieval_time
            }
        else:
            timing_logger.info(f"⚡ STREAM NO RAG - skipping retrieval")
            docs = []
            yield {"type": "documents", "data": [], "count": 0}

        # 2. Build system message
        msg_start = time.time()
        system_msg = make_system_message(prompt_template, docs, user_files or [])
        msg_time = time.time() - msg_start
        timing_logger.info(f"📝 STREAM MESSAGE BUILD: {msg_time:.3f}s, msg_len={len(system_msg)}")

        # 3. Model loading
        yield {"type": "status", "message": "Loading models..."}
        model_start = time.time()
        langchain_models.pull_model(embedding_model)
        langchain_models.pull_model(llm_model)
        model_time = time.time() - model_start
        timing_logger.info(f"🔧 STREAM MODEL LOADING: {model_time:.3f}s")

        # 4. Start LLM streaming
        yield {"type": "status", "message": "Generating response..."}
        llm_start = time.time()
        timing_logger.info(f"🧠 STREAM LLM START: model={llm_model}")
        
        # Try primary LLM query first
        try:
            timing_logger.info(f"🔧 CALLING langchain_llm.query_llm with model: {llm_model}")
            tokens = langchain_llm.query_llm(llm_model, query, system_msg)
            timing_logger.info(f"✅ GOT TOKENS from query_llm: {type(tokens)}")
            
            model_source, _ = langchain_utils.parse_model_name(llm_model)
            
            answer_tokens = []
            timing_logger.info(f"🔧 CREATING token_generator for model_source: {model_source}")
            token_generator = _async_token_generator(tokens, model_source)
            
            timing_logger.info(f"🔧 STARTING async iteration over tokens")
            async for chunk in token_generator:
                token_content = chunk
                answer_tokens.append(token_content)
                
                yield {
                    "type": "token",
                    "content": token_content,
                    "partial_answer": "".join(answer_tokens)
                }
            
            llm_time = time.time() - llm_start
            timing_logger.info(f"🧠 STREAM LLM SUCCESS (primary): {llm_time:.3f}s, answer_len={len(''.join(answer_tokens))}")
            
        except Exception as e:
            # Clear any partial response and start fresh with fallback
            timing_logger.error(f"⚠️ STREAM LLM FALLBACK ERROR: {str(e)} - Type: {type(e)} - Model: {llm_model}")
            
            # Clear previous response and show fallback status
            yield {"type": "clear_response", "message": "Primary LLM failed, using fallback..."}
            yield {"type": "status", "message": f"Using fallback LLM mode (Primary failed: {str(e)[:50]}...)..."}
            
            fallback_start = time.time()
            model_source, model_name = langchain_utils.parse_model_name(llm_model)

            if model_source == "openai":
                chat_llm = ChatOpenAI(
                    model=model_name,
                    temperature=0.0,
                    streaming=True,
                )
            elif model_source == "google":
                chat_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.0,
                    streaming=True,
                )
            elif model_source == "stanford":
                chat_llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=config.stanford_api_key,
                    openai_api_base=config.stanford_base_url,
                    temperature=0.0,
                    streaming=True,
                )
            else:
                raise RuntimeError(f"Unsupported fallback model source: {model_source}")

            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=query),
            ]

            # Stream from ChatOpenAI
            token_stream = chat_llm.stream(messages)
            answer_tokens = []
            
            for chunk in token_stream:
                if hasattr(chunk, 'content'):
                    token_content = chunk.content
                else:
                    token_content = str(chunk)
                
                if token_content:  # Only yield non-empty tokens
                    answer_tokens.append(token_content)
                    yield {
                        "type": "token",
                        "content": token_content,
                        "partial_answer": "".join(answer_tokens)
                    }

            llm_time = time.time() - llm_start
            fallback_time = time.time() - fallback_start
            timing_logger.info(f"🧠 STREAM LLM SUCCESS (fallback): fallback={fallback_time:.3f}s, total_llm={llm_time:.3f}s, answer_len={len(''.join(answer_tokens))}")

        # 5. Send final summary
        total_time = time.time() - start_time
        timing_logger.info(f"🌊 ANSWER QUERY STREAM END: total={total_time:.3f}s")
        
        yield {
            "type": "complete",
            "final_answer": "".join(answer_tokens),
            "documents": doc_info if use_rag else [],
            "total_time": total_time,
            "llm_time": llm_time
        }

    except Exception as e:
        timing_logger.error(f"❌ STREAM ERROR: {str(e)}")
        yield {
            "type": "error",
            "message": str(e),
            "timestamp": time.time()
        }


async def _async_token_generator(tokens, model_source):
    """Convert synchronous token generator to async for streaming."""
    for token in tokens:
        content = None
        
        # Debug: Log token details for Stanford API
        if model_source == "stanford":
            timing_logger.info(f"🔍 STANFORD TOKEN DEBUG: type={type(token)}, token={repr(token)}, hasattr_content={hasattr(token, 'content')}, hasattr_text={hasattr(token, 'text')}")
            if hasattr(token, 'content'):
                timing_logger.info(f"   content={repr(getattr(token, 'content', None))}")
            if hasattr(token, 'text'):
                timing_logger.info(f"   text={repr(getattr(token, 'text', None))}")
        
        if model_source in ["openai", "stanford", "google"]:
            # Handle different token formats from different providers
            if hasattr(token, 'content') and token.content is not None:
                content = str(token.content)
            elif hasattr(token, 'text') and token.text is not None:
                content = str(token.text)
            elif token is not None:
                content = str(token)
        else:
            # For Ollama and other providers
            if token is not None:
                content = str(token)
        
        # Only yield non-empty, non-None tokens
        if content and content.strip():
            yield content
        
        # Allow other coroutines to run
        await asyncio.sleep(0)
