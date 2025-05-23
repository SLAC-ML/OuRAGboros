import io
import time

import streamlit as st

import lib.config as config
import lib.streamlit.nav as nav
import lib.streamlit.session_state as ss
import lib.rag_service as service

# Streamlit page configuration
st.set_page_config(
    page_title="OuRAGboros",
    page_icon=":snake:",
    layout="wide",
)

# Initialize navigation bar and session state
nav.pages()
available_llms, available_embeddings = ss.init()

st.title(":snake: LangChain OuRAGboros")

# Sidebar: search configuration
with st.sidebar:
    st.header("Search Configuration")
    st.toggle(
        "Use OpenSearch",
        key=ss.StateKey.USE_OPENSEARCH,
        help=f"Requires an OpenSearch instance running at {config.opensearch_base_url}. "
             "If this toggle is off, all documents are retrieved from an in-memory vector "
             "store which is lost when the application terminates.",
    )
    st.selectbox(
        "Embedding model:",
        available_embeddings,
        key=ss.StateKey.EMBEDDING_MODEL
    )
    st.selectbox(
        "LLM:",
        available_llms,
        key=ss.StateKey.LLM_MODEL
    )
    st.slider(
        "Set the document match score threshold:",
        min_value=0.0,
        max_value=2.0,
        help="Score is computed using cosine similarity plus 1 to ensure a non-negative score.",
        key=ss.StateKey.QUERY_RESULT_SCORE_INF
    )
    st.slider(
        "Set the maximum retrieved documents:",
        min_value=1,
        max_value=50,
        step=1,
        help="Sets an upper bound on the number of documents to return.",
        key=ss.StateKey.MAX_DOCUMENT_COUNT
    )
    st.text_area(
        "Model Prompt",
        height=250,
        key=ss.StateKey.LLM_PROMPT
    )

# Chat input: user query and optional file uploads
search_query = st.chat_input(
    "Enter a search query",
    accept_file=True,
    file_type=config.text_file_types,
)

if search_query:
    if search_query.text:
        st.session_state[ss.StateKey.SEARCH_QUERY] = search_query.text
        st.session_state[ss.StateKey.USER_CONTEXT] = [
            (ctx_file.name, io.StringIO(ctx_file.getvalue().decode('utf-8')).read())
            for ctx_file in search_query.files
        ]

# Function to render retrieved source documents

def _render_source_docs(docs):
    for i, (doc, score) in enumerate(docs):
        if i:
            st.divider()
        st.subheader(ss.document_file_name(doc))
        st.download_button(
            label="Download as text",
            data=doc.page_content,
            file_name=ss.document_file_name(doc),
            mime="text/plain",
            key=doc.id,
            on_click="ignore"
        )
        st.markdown(f"**Score:** {score}")
        st.markdown('**Document Text:**')
        st.code(
            f"{doc.page_content[:1000]}" +
            ('... [download file to see more]' if len(doc.page_content) > 100 else ''),
            language=None,
            line_numbers=True,
            wrap_lines=True,
        )

# Main chat loop: perform RAG and display results
if (
    st.session_state.get(ss.StateKey.SEARCH_QUERY) and
    st.session_state.get(ss.StateKey.LLM_MODEL)
):
    # Echo user query
    with st.chat_message("user"):
        st.text(st.session_state[ss.StateKey.SEARCH_QUERY])

    # Call into our shared RAG service
    with st.spinner("Running retrieval + LLM…"):
        answer, docs = service.answer_query(
            query=st.session_state[ss.StateKey.SEARCH_QUERY],
            embedding_model=st.session_state[ss.StateKey.EMBEDDING_MODEL],
            llm_model=st.session_state[ss.StateKey.LLM_MODEL],
            k=st.session_state[ss.StateKey.MAX_DOCUMENT_COUNT],
            score_threshold=st.session_state[ss.StateKey.QUERY_RESULT_SCORE_INF],
            use_opensearch=st.session_state[ss.StateKey.USE_OPENSEARCH],
            prompt_template=st.session_state[ss.StateKey.LLM_PROMPT],
            user_files=st.session_state.get(ss.StateKey.USER_CONTEXT, []),
        )

    # Display the LLM’s answer
    with st.chat_message("ai"):
        st.text(answer)

    # Store & render source docs
    st.session_state[ss.StateKey.RAG_DOCS] = docs
    if docs:
        singular = len(docs) == 1
        with st.expander(f"Source document{'s' if not singular else ''}"):
            _render_source_docs(docs)

    # Allow user to download session
    st.download_button(
        "Download session data",
        ss.dump_session_state(),
        file_name=f"ouragboros_session_data_{time.time()}.json",
        mime="application/json",
    )
