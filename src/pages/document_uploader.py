from io import StringIO
import hashlib
import time

import opensearchpy.exceptions
import streamlit as st
from opensearchpy import OpenSearch

import lib.config as config
import lib.nav as nav
import lib.langchain_impl as langchain_impl

st.set_page_config(
    page_title='OpenSearch Document Upload',
    page_icon=':page_facing_up:',
    layout='wide',
)
nav.pages()

st.title(':page_facing_up: OpenSearch Document Upload')

desired_file_extension = '.md'
document_search_glob = f'*{desired_file_extension}'

with st.sidebar:
    st.header('Search Configuration')
    embedding_model = st.selectbox(
        'Select an embedding model:',
        langchain_impl.get_available_embeddings(),
        index=0,
    )

uploaded_files = st.file_uploader(
    'Upload source files to be added to the application knowledgebase.',
    accept_multiple_files=True,
    type=['txt', 'md'],
)

# Upload documents to OpenSearch
#
if len(uploaded_files) and st.button('Upload Files'):
    # Create OpenSearch index if it doesn't already exist
    #
    try:
        st.text('Ensuring OpenSearch index existence...')
        vector_size = len(langchain_impl.get_embedding(embedding_model).embed_query('hi'))

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

    vector_store = langchain_impl.opensearch_doc_vector_store(embedding_model)

    for uploaded_file in uploaded_files:
        st.text(f'Uploading {uploaded_file.name}...')
        file_content = StringIO(uploaded_file.getvalue().decode('utf-8')).read()
        file_sha1 = hashlib.sha1(uploaded_file.getbuffer()).hexdigest()
        vector_store.add_texts(
            texts=[
                file_content
            ],
            metadatas=[
                {
                    'embedding_model': embedding_model,
                    'sha1': file_sha1,
                    'size': uploaded_file.size,
                    'source': uploaded_file.name,
                    'uploaded': int(time.time())
                }
            ],
            ids=[f'{file_sha1}_{embedding_model}']
        )
    st.write('Done!')
