from io import BytesIO
import hashlib
import time

import opensearchpy.exceptions
import streamlit as st
from langchain_core.vectorstores import VectorStore
from opensearchpy import OpenSearch

import lib.config as config
import lib.nav as nav
import lib.langchain_impl as langchain_impl

st.set_page_config(
    page_title='OpenSearch Document Embedding',
    page_icon=':page_facing_up:',
    layout='wide',
)
nav.pages()

st.title(':page_facing_up: OpenSearch Document Embedding')

desired_file_extension = '.md'
document_search_glob = f'*{desired_file_extension}'


def _ensure_opensearch_index(embedding_model_name):
    try:
        vector_size = len(
            langchain_impl.get_embedding(embedding_model_name).embed_query('hi')
        )

        opensearch_client = OpenSearch([
            config.opensearch_url
        ])
        opensearch_client.indices.create(
            index=config.opensearch_index,
            body=config.opensearch_index_settings(vector_size=vector_size),
        )
    except opensearchpy.exceptions.RequestError as e:
        if e.status_code != 400:
            raise e


def _upload_text_to_vector_store(
        vs: VectorStore,
        embedding_model_name: str,
        text_file_bytes: BytesIO,
        text_file_name: str,
):
    text_bytes = text_file_bytes.getvalue()
    file_content = text_bytes.decode('utf-8')
    file_sha1 = hashlib.sha1(text_file_bytes.getbuffer()).hexdigest()

    metadata = {
        'embedding_model': embedding_model_name,
        'sha1': file_sha1,
        'size': len(text_bytes),
        'source': text_file_name,
        'uploaded': int(time.time())
    }

    vs.add_texts(
        texts=[
            file_content
        ],
        metadatas=[metadata],
        ids=[f'{file_sha1}_{embedding_model_name}']
    )


with st.sidebar:
    st.header('Search Configuration')
    use_opensearch = st.toggle(
        'Use OpenSearch',
        help=f'Requires an OpenSearch instance running at {config.opensearch_url}. If '
             f'this toggle is off, all documents are stored in-memory and are lost when '
             f'the application terminates.'
    )
    embedding_model = st.selectbox(
        'Select an embedding model:',
        langchain_impl.get_available_embeddings(),
        index=0,
    )

# Upload text documents to OpenSearch
#
uploaded_text_files = st.file_uploader(
    'Upload raw text files to be added to the application knowledgebase.',
    accept_multiple_files=True,
    type=['txt', 'md', '.tex'],
)
if len(uploaded_text_files) and st.button('Embed Text'):
    # Create OpenSearch index if it doesn't already exist
    #
    with st.spinner(f'Pulling `{embedding_model}`...'):
        langchain_impl.pull_model(embedding_model)

    if use_opensearch:
        st.text('Ensuring OpenSearch index existence...')
        _ensure_opensearch_index(embedding_model)
        vector_store = langchain_impl.opensearch_doc_vector_store(embedding_model)
    else:
        vector_store = langchain_impl.get_in_memory_vector_store(embedding_model)

    text_upload_bar = st.empty()


    def text_upload_progress(i: int, filename: str, end: int):
        with text_upload_bar.container():
            st.progress(i / end, f'Embedding {filename} [{i}/{end}]...')


    for k, uploaded_text_file in enumerate(uploaded_text_files):
        text_upload_progress(k + 1, uploaded_text_file.name, len(uploaded_text_files))

        _upload_text_to_vector_store(
            vs=vector_store,
            embedding_model_name=embedding_model,
            text_file_bytes=uploaded_text_file,
            text_file_name=uploaded_text_file.name,
        )

    st.write('Done!')

# Upload PDFs to OpenSearch
#
pdf_doc = st.file_uploader(
    'Upload PDF document to be parsed and added to the application knowledgebase.',
    type=['pdf'],
)

if pdf_doc and st.button('Embed PDF'):
    import lib.pdf as pdf

    with st.spinner(f'Pulling `{embedding_model}` embeddings...'):
        langchain_impl.pull_model(embedding_model)

    # Create OpenSearch index if it doesn't already exist
    #
    if use_opensearch:
        st.text('Ensuring OpenSearch index existence...')
        _ensure_opensearch_index(embedding_model)
        vector_store = langchain_impl.opensearch_doc_vector_store(embedding_model)
    else:
        vector_store = langchain_impl.get_in_memory_vector_store(embedding_model)

    pdf_upload_bar = st.empty()


    def pdf_upload_progress(i: int, filename: str, end: int):
        with pdf_upload_bar.container():
            st.progress(i / end, f'Embedding {filename} [{i}/{end}]...')


    pdf_extractor = pdf.NougatExtractor()
    for k, (txt_bytes, txt_name, pages) in enumerate(
            pdf_extractor.extract_text(pdf_doc, pdf_doc.name)
    ):
        pdf_upload_progress(k + 1, txt_name, len(pages))

        _upload_text_to_vector_store(
            vs=vector_store,
            embedding_model_name=embedding_model,
            text_file_bytes=txt_bytes,
            text_file_name=txt_name,
        )

    st.write('Done!')
