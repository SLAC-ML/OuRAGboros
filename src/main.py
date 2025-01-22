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
        score_threshold: float = 0.2
):
    vector_store = langchain_impl.vectorize_md_docs(doc_path)
    return [
        d for d in vector_store.similarity_search_with_score(query, k=k)
        if d[1] >= score_threshold
    ]


root_doc_path = st.text_input(
    label='Root document path:',
    help='This folder that will be recursively indexed for RAG-based search.',
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

st.header(f'Indexable Files [{mimetypes.types_map[desired_file_extension]}]', help='')
st.text('These files will be indexed for RAG generation.')
st.dataframe(doc_df, use_container_width=True)

query_result_score_inf = st.number_input(
    'Set the document match score threshold:',
    value=0.4,
    min_value=0.0,
    max_value=1.0,
)

llm_model = st.selectbox(
    'Select an LLM Model:',
    langchain_impl.get_models(),
    index=0,
)

rag_query = st.text_input(
    label='Enter a RAG search query:',
    value='',
    key='rag_query',
)

if rag_query and llm_model and query_result_score_inf:
    with st.spinner('Pulling model...'):
        langchain_impl.pull_model(llm_model)

    st.text('Performing document search...')
    matches = perform_document_retrieval(
        root_doc_path,
        rag_query,
        k=3,
        score_threshold=query_result_score_inf
    )

    if not len(matches):
        raise ValueError(
            'No document matches found. Try a new query, or lower the score threshold.'
        )
    else:
        st.text(f'Found {len(matches)} matches.')

    st.text('Getting LLM response...')

    context = '\n'.join([
        doc.page_content for doc, score in matches
    ])
    st.write(langchain_impl.ask_llm(rag_query, context))

    st.divider()
    n_doc_matches = len(matches)
    if n_doc_matches:
        if n_doc_matches == 1:
            st.text('Top document match:')
        else:
            st.text(f'Top {n_doc_matches} document matches:')

        for doc, score in matches:
            st.subheader(Path(doc.metadata['source']).name)
            st.markdown(f'**File Path:** {doc.metadata['source']}')
            st.markdown(f'**Score:** {score}')

            st.markdown('#### File Contents:')
            st.code(doc.page_content, language=None, line_numbers=True, wrap_lines=True)
            st.divider()
    else:
        st.warning('No document matches found.')
