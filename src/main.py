from pathlib import Path

import streamlit as st

import lib.config as config
import lib.langchain_impl as langchain_impl
import lib.nav as nav

st.set_page_config(
    page_title='OuRAGbourous',
    page_icon=':snake:',
    layout='wide',
)
nav.pages()

st.title(':snake: LangChain OuRAGborous')

# st.write(st.session_state)

# Initialize session state
#
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'search_query' not in st.session_state:
    st.session_state.search_query = ''
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = ''
if 'root_doc_path' not in st.session_state:
    st.session_state.root_doc_path = config.default_root_doc_path

def perform_document_retrieval(
        doc_path: str,
        query: str,
        k=3,
        score_threshold: float = 0.2,
        model: str = None
):
    vector_store = langchain_impl.vectorize_md_docs(doc_path, embedding_model=model)
    return [
        d for d in vector_store.similarity_search_with_score(query, k=k)
        if d[1] >= score_threshold
    ]


with st.sidebar:
    st.header('Search Configuration')
    embeddings_model = st.selectbox(
        'Select an embedding model:',
        langchain_impl.get_embedding_models(),
        index=0,
    )
    llm_model = st.selectbox(
        'Select an LLM model:',
        langchain_impl.get_llm_models(),
        index=0,
    )
    query_result_score_inf = st.slider(
        'Set the document match score threshold:',
        value=0.3,
        min_value=0.01,
        max_value=1.0,
    )

search_query = st.chat_input('Enter a search query')

@st.fragment
def _render_source_docs(docs):
    """
    Renders source documents used to generate LLM context.

    We put this function inside its own fragment so that download links don't trigger a
    full page refresh.

    :param docs:
    :return:
    """
    for i, (doc, score) in enumerate(docs):
        if i:
            st.divider()
        st.subheader(Path(doc.metadata['source']).name)
        st.markdown(f'**File Path:** {doc.metadata['source']}')
        st.markdown(f'**Score:** {score}')
        st.download_button(
            label='Download as text',
            data=doc.page_content,
            file_name=Path(doc.metadata['source']).name,
            mime='text/plain',
        )

        st.markdown('#### File Contents:')
        st.code(
            f'{doc.page_content[:-100]}...(download file to see more)',
            language=None,
            line_numbers=True,
            wrap_lines=True,
        )

# Save search query if a new one was provided
#
if search_query:
    st.session_state.search_query = search_query

# Main page content
#
if st.session_state.search_query and llm_model and query_result_score_inf:
    with st.chat_message('user'):
        st.text(st.session_state.search_query)

    with st.spinner('Pulling model...'):
        langchain_impl.pull_model(llm_model)

    with st.chat_message('ai'):
        st.text('Searching for relevant documentation...')

        matches = perform_document_retrieval(
            st.session_state.root_doc_path,
            st.session_state.search_query,
            k=3,
            score_threshold=query_result_score_inf,
            model=embeddings_model
        )
        st.session_state.documents = matches

        singular_match = len(st.session_state.documents) == 1
        if len(st.session_state.documents):
            st.text('Found {} match{}. '.format(
                len(st.session_state.documents),
                '' if singular_match else 'es'
            ))

            context = '\n'.join([
                doc.page_content for doc, score in st.session_state.documents
            ])

            st.session_state.llm_response = langchain_impl.ask_llm(
                st.session_state.search_query,
                context,
                llm_model=llm_model,
            )
            st.write_stream(st.session_state.llm_response)

            with st.expander(f'Source document{'' if singular_match else 's'}'):
                _render_source_docs(st.session_state.documents)
        else:
            st.warning(
                'No document matches found. Try a new query, or lower the score threshold.'
            )
