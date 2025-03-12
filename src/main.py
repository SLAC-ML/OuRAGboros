import io
import itertools
import time

import streamlit as st
import langchain_community.vectorstores.opensearch_vector_search as os_vs

import lib.config as config

import lib.streamlit.nav as nav
import lib.streamlit.session_state as ss

import lib.langchain.llm as langchain_llm
import lib.langchain.models as langchain_models
import lib.langchain.opensearch as langchain_opensearch

st.set_page_config(
    page_title="OuRAGboros",
    page_icon=":snake:",
    layout="wide",
)

# Initialize navigation bar
#
nav.pages()

# Initialize session state
#
available_llms, available_embeddings = ss.init()

st.title(":snake: LangChain OuRAGboros")



def perform_document_retrieval(
        query: str,
        model: str,
        k: int,
        score_threshold: float,
        use_opensearch_vectorstore: bool,
):
    """
    Retrieves a list of Document objects that correspond to the provided search query
    by executing a cosine similarity search.

    :param query:
    :param k:
    :param score_threshold:
    :param model:
    :param use_opensearch_vectorstore:
    :return:
    """
    vs = ss.get_vector_store(use_opensearch_vectorstore, model)
    if use_opensearch_vectorstore:
        st.text("Ensuring OpenSearch index existence...")
        langchain_opensearch.ensure_opensearch_index(model)

        return vs.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=score_threshold,
            search_type=os_vs.SCRIPT_SCORING_SEARCH,
            # See: https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/#spaces
            #
            space_type="cosinesimil"
        )
    else:
        # LangChain's in-memory vector store uses cosine similarity by default.
        # See: https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html#langchain_core.vectorstores.in_memory.InMemoryVectorStore
        #
        # We add 1 to the score to keep formatting consistent with OpenSearch
        #
        return [
            (d, s + 1) for (d, s) in vs.similarity_search_with_score(query, k=k)
            if s + 1 >= score_threshold
        ]


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
        help="Score is computed using cosine similarity plus 1 to ensure a non-negative "
             "score.",
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

search_query = st.chat_input(
    "Enter a search query",
    accept_file=True,
    file_type=config.text_file_types,
)

# Save search query into session state since the chat widgets cannot be set via
# `st.session_state`.
#
if search_query:
    if search_query.text:
        st.session_state[ss.StateKey.SEARCH_QUERY] = search_query.text

        st.session_state[ss.StateKey.USER_CONTEXT] = [
            (ctx_file.name, io.StringIO(ctx_file.getvalue().decode('utf-8')).read())
            for ctx_file in search_query.files
        ]


# @st.fragment
def _render_source_docs(docs):
    """
    Renders source documents used to generate LLM system_message.

    We put this function inside its own fragment so that download links don't trigger a
    full page refresh.

    :param docs:
    :return:
    """
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
        st.code('{}{}'.format(
            doc.page_content[:1000],
            '... [download file to see more]' if len(doc.page_content) > 100 else ''
        ),
            language=None,
            line_numbers=True,
            wrap_lines=True,
        )


# Main page content
#
if (
        st.session_state[ss.StateKey.SEARCH_QUERY] and
        st.session_state[ss.StateKey.LLM_MODEL]
):
    with st.chat_message("user"):
        st.text(st.session_state[ss.StateKey.SEARCH_QUERY])

    with st.spinner("Loading `{}` embeddings...".format(
            st.session_state[ss.StateKey.EMBEDDING_MODEL]
    )):
        langchain_models.pull_model(st.session_state[ss.StateKey.EMBEDDING_MODEL])

    with st.spinner("Loading `{}` LLM...".format(
            st.session_state[ss.StateKey.LLM_MODEL]
    )):
        langchain_models.pull_model(st.session_state[ss.StateKey.LLM_MODEL])

    with ((st.chat_message("ai"))):
        st.text("Searching knowledge base for relevant documentation...")

        matches = perform_document_retrieval(
            st.session_state[ss.StateKey.SEARCH_QUERY],
            st.session_state[ss.StateKey.EMBEDDING_MODEL],
            k=st.session_state[ss.StateKey.MAX_DOCUMENT_COUNT],
            score_threshold=st.session_state[ss.StateKey.QUERY_RESULT_SCORE_INF],
            use_opensearch_vectorstore=st.session_state[ss.StateKey.USE_OPENSEARCH],
        )
        st.session_state[ss.StateKey.RAG_DOCS] = matches

        singular_match = len(st.session_state[ss.StateKey.RAG_DOCS]) == 1

        if len(st.session_state[ss.StateKey.RAG_DOCS]):
            st.text("Found {} document match{}.".format(
                len(st.session_state[ss.StateKey.RAG_DOCS]),
                "" if singular_match else "es"
            ))
        else:
            st.warning(
                "No document matches found using embedding model "
                f"`{st.session_state[ss.StateKey.EMBEDDING_MODEL]}`. Try a new query or "
                "lower the score threshold for a more contextually relevant response."
            )

        context = [
            *[(doc.id, doc.page_content)
              for doc, score in st.session_state[ss.StateKey.RAG_DOCS]],
            *st.session_state[ss.StateKey.USER_CONTEXT]
        ]

        if len(context):
            st.session_state[ss.StateKey.SYSTEM_MESSAGE] = "\n".join([
                st.session_state[ss.StateKey.LLM_PROMPT],
                f"\nSources: \n{"\n".join([
                    f"---{source[0]}---\n {source[1]}" for source in context
                ])}"
            ])
        else:
            st.session_state[ss.StateKey.SYSTEM_MESSAGE] = st.session_state[
                ss.StateKey.LLM_PROMPT
            ]

        # Query LLM and output text in real time. Once all text has been printed, we
        # save the entire string to session state.
        #
        llm_response_str, llm_response_out = itertools.tee(langchain_llm.query_llm(
            st.session_state[ss.StateKey.LLM_MODEL],
            st.session_state[ss.StateKey.SEARCH_QUERY],
            st.session_state[ss.StateKey.SYSTEM_MESSAGE]
        ), 2)
        st.write_stream(llm_response_out)  # Output to user
        st.session_state[ss.StateKey.LLM_RESPONSE] = "".join(llm_response_str)  # Save

        if len(st.session_state[ss.StateKey.RAG_DOCS]):
            with st.expander(f"Source document{"" if singular_match else "s"}"):
                _render_source_docs(st.session_state[ss.StateKey.RAG_DOCS])

        st.download_button(
            "Download session data",
            ss.dump_session_state(),
            file_name=f"ouragboros_{time.time()}.json",
            on_click="ignore",
            mime="application/json",
        )
