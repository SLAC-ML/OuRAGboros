import json

import streamlit as st

import lib.langchain.embeddings as langchain_embeddings
import lib.langchain.llm as langchain_llm

import lib.config as config


class StateKey(str):
    """
    Contains all configuration keys used for a particular application run (e.g., limit on
    number of vector store documents retrieved, LLM used, etc.)
    """
    RAG_DOCS = "rag_documents"
    EMBEDDING_CHUNK_OVERLAP = "embedding_chunk_overlap"
    EMBEDDING_CHUNK_SIZE = "embedding_chunk_size"
    EMBEDDING_MODEL = "embedding_model"
    LLM_MODEL = "llm_model"
    LLM_PROMPT = "llm_prompt"
    LLM_RESPONSE = "llm_response"
    MAX_DOCUMENT_COUNT = "max_document_count"
    QUERY_RESULT_SCORE_INF = "query_result_score_inf"
    SEARCH_QUERY = "search_query"
    SYSTEM_MESSAGE = "system_message"
    USE_OPENSEARCH = "use_opensearch"
    USER_CONTEXT = "user_context"
    VECTOR_STORE = "vector_store"


def init() -> tuple[list[str], list[str]]:
    """
    Initialize session state for the application. For more information about how Streamlit
    uses session state, see:

    https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state#serializable-session-state

    :return:
    """
    available_embeddings = langchain_embeddings.get_available_embeddings()
    available_llms = langchain_llm.get_available_llms()

    default_session_state = [
        (StateKey.RAG_DOCS, []),
        (StateKey.EMBEDDING_CHUNK_OVERLAP, 20),
        (StateKey.EMBEDDING_CHUNK_SIZE, 4000),
        (StateKey.EMBEDDING_MODEL, available_embeddings[0]),
        (StateKey.LLM_MODEL, available_llms[0]),
        (StateKey.LLM_PROMPT, config.default_prompt),
        (StateKey.LLM_RESPONSE, ""),
        (StateKey.MAX_DOCUMENT_COUNT, 5),
        (StateKey.QUERY_RESULT_SCORE_INF, 0.7),
        (StateKey.SEARCH_QUERY, ""),
        (StateKey.USER_CONTEXT, []),
        (StateKey.USE_OPENSEARCH, config.prefer_opensearch),
        (StateKey.VECTOR_STORE, None),
    ]
    for state_var, state_val in default_session_state:
        if state_var not in st.session_state:
            st.session_state[state_var] = state_val

    return available_llms, available_embeddings

def dump_session_state():
    state = {}

    for key in [
        StateKey.EMBEDDING_CHUNK_OVERLAP,
        StateKey.EMBEDDING_CHUNK_SIZE,
        StateKey.EMBEDDING_MODEL,
        StateKey.LLM_MODEL,
        StateKey.LLM_PROMPT,
        StateKey.LLM_RESPONSE,
        StateKey.MAX_DOCUMENT_COUNT,
        StateKey.QUERY_RESULT_SCORE_INF,
        StateKey.SEARCH_QUERY,
        StateKey.USER_CONTEXT,
        StateKey.USE_OPENSEARCH
    ]:
        state[key] = st.session_state[key]

    state[StateKey.RAG_DOCS] = [
        (doc.id, doc.page_content) for doc in st.session_state[StateKey.RAG_DOCS]
    ]

    return json.dumps(state)
