# lib/rag_service.py
import itertools
from typing import List, Dict, Tuple

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


def perform_document_retrieval(
    query: str,
    embedding_model: str,
    k: int,
    score_threshold: float,
    use_opensearch_vectorstore: bool,
    knowledge_base: str = "default",
) -> List[Tuple[Document, float]]:
    vs = ss.get_vector_store(use_opensearch_vectorstore, embedding_model, knowledge_base)
    if use_opensearch_vectorstore:
        langchain_opensearch.ensure_opensearch_index(embedding_model, knowledge_base)
        return vs.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=score_threshold,
            search_type=SCRIPT_SCORING_SEARCH,
            space_type="cosinesimil",
        )
    else:
        return [
            (d, s + 1)
            for (d, s) in vs.similarity_search_with_score(query, k=k)
            if s + 1 >= score_threshold
        ]


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
) -> Tuple[str, List[Tuple[Document, float]]]:
    """
    Returns (full_answer_text, retrieved_docs), streaming is only in UI.
    """
    # 1. retrieve
    if use_rag:
        docs = perform_document_retrieval(
            query, embedding_model, k, score_threshold, use_opensearch, knowledge_base
        )
    else:
        # if not using RAG, return an empty docs list
        docs = []

    # 2. build the base system message
    system_msg = make_system_message(prompt_template, docs, user_files or [])

    # 3. if there *is* prior chat‐history, stitch it in
    if history:
        # turn [{"role":"user","content":"…"}, …] into text
        hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        system_msg += "\n\nConversation history:\n" + hist_text

    # 4. ensure models are loaded
    langchain_models.pull_model(embedding_model)
    langchain_models.pull_model(llm_model)

    # 5. query
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
    except Exception:
        # instantiate a streaming chat LLM
        model_source, model_name = langchain_utils.parse_model_name(llm_model)

        if model_source == "openai":
            chat_llm = ChatOpenAI(
                model_name=llm_model,
                temperature=0.0,
                streaming=True,
            )
        elif model_source == "google":
            chat_llm = ChatGoogleGenerativeAI(
                model=model_name,
                # google_api_key=config.google_api_key,
                temperature=0.0,
                streaming=True,
            )
        elif model_source == "stanford":
            # Stanford API uses OpenAI-compatible interface
            chat_llm = ChatOpenAI(
                model=model_name,
                openai_api_key=config.stanford_api_key,
                openai_api_base=config.stanford_base_url,
                temperature=0.0,
                streaming=True,
            )
        else:
            raise RuntimeError(f"Unsupported fallback model source: {model_source}")

        # wrap your system prompt and user query
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=query),
        ]

        # get a token generator and tee it
        token_stream = chat_llm.stream(messages)
        tokens, tokens_for_save = itertools.tee(token_stream, 2)

        # collect the full answer
        if model_source in ["google", "openai", "stanford"]:
            answer = "".join(chunk.content for chunk in tokens_for_save)
        else:
            answer = "".join(tokens_for_save)

    return answer, docs
