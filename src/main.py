import os
from pathlib import Path
import mimetypes

import streamlit as st
import pandas as pd

import langchain_impl

st.set_page_config(layout='wide', page_title='OuRAGbourous')
st.title(':snake: LangChain OuRAGborous :snake:')
st.session_state.retrieved_docs = []

default_root_doc_path = (
    '/work/jonathan/stanford/stanford-academics/StanfordNotes/Research/SLAC LLM (Darve Winter 2025)'
)
desired_file_extension = '.md'
document_search_glob = f'*{desired_file_extension}'


def perform_document_retrieval(
        doc_path: str,
        query: str,
        k=3,
        score_threshold: float = 0.2,
        model: str = None
):
    vector_store = langchain_impl.vectorize_md_docs(doc_path, ollama_model=model)
    return [
        d for d in vector_store.similarity_search_with_score(query, k=k)
        if d[1] >= score_threshold
    ]


root_doc_path = st.text_input(
    label='Root document path:',
    help='This folder that will be recursively indexed for search.',
    value=default_root_doc_path,
)

if not os.path.exists(root_doc_path):
    raise ValueError('Selected root document path does not exist.')

docs = list(Path(root_doc_path).rglob(document_search_glob))
doc_df = pd.DataFrame([
    (f'.{str(doc).removeprefix(default_root_doc_path)}',
     doc.stat().st_size
     ) for doc in docs
], columns=['Name', 'Size [bytes]'])

st.text('Found the following documents:')
st.dataframe(doc_df, use_container_width=True)

with st.sidebar:
    st.title('Configuration')
    embeddings_model = st.selectbox(
        'Select an embedding model:',
        langchain_impl.get_models(),
        index=0,
    )
    llm_model = st.selectbox(
        'Select an LLM model:',
        langchain_impl.get_models(),
        index=0,
    )
    query_result_score_inf = st.slider(
        'Set the document match score threshold:',
        value=0.4,
        min_value=0.0,
        max_value=1.0,
    )

rag_query = st.chat_input('Enter a RAG search query:')

if rag_query and llm_model and query_result_score_inf:
    with st.chat_message('user'):
        st.text(rag_query)

    with st.spinner('Pulling model...'):
        langchain_impl.pull_model(llm_model)

    with st.chat_message('ai'):
        st.text('Searching for relevant documentation...')
        matches = perform_document_retrieval(
            root_doc_path,
            rag_query,
            k=3,
            score_threshold=query_result_score_inf,
            model=embeddings_model
        )

        singular_match = len(matches) == 1
        if len(matches):
            st.text(f'Found {len(matches)} match{'' if singular_match else 'es'}. '
                    f'Generating LLM response.')

            context = '\n'.join([
                doc.page_content for doc, score in matches
            ])

            st.write_stream(
                langchain_impl.ask_llm(rag_query, context, ollama_model=llm_model),
            )

            with st.expander(f'Source document{'' if singular_match else 's'}'):
                for i, (doc, score) in enumerate(matches):
                    if i:
                        st.divider()
                    st.subheader(Path(doc.metadata['source']).name)
                    st.markdown(f'**File Path:** {doc.metadata['source']}')
                    st.markdown(f'**Score:** {score}')

                    st.markdown('#### File Contents:')
                    st.code(doc.page_content, language=None, line_numbers=True,
                            wrap_lines=True)
        else:
            st.warning(
                'No document matches found. Try a new query, or lower the score threshold.'
            )
