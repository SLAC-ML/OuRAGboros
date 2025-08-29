# lib/rag_service.py
import itertools
from typing import List, Dict, Tuple
import time
import logging

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
