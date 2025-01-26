from io import StringIO
import hashlib

from opensearchpy import OpenSearch
import streamlit as st
from langchain_community.vectorstores import (
    OpenSearchVectorSearch
)

import lib.config as config
import lib.nav as nav
import lib.langchain_impl as langchain_impl
from lib.langchain_impl import get_embedding

st.set_page_config(
    page_title='Document Uploader',
    page_icon=':page_facing_up:',
    layout='wide',
)
nav.pages()

st.title(':page_facing_up: Document Upload')

desired_file_extension = '.md'
document_search_glob = f'*{desired_file_extension}'

with st.sidebar:
    st.header('Search Configuration')
    embeddings_model = st.selectbox(
        'Select an embedding model:',
        langchain_impl.get_embedding_models(),
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
    st.text('Ensuring OpenSearch index existence...')
    opensearch_client = OpenSearch([
        config.opensearch_url
    ])
    opensearch_client.indices.create(
        index=config.opensearch_index,
        body=config.opensearch_index_settings,
        ignore=400
    )

    embeddings = get_embedding(embeddings_model)
    vector_search = OpenSearchVectorSearch(
        index_name=config.opensearch_index,
        opensearch_url=config.opensearch_url,
        embedding_function=embeddings,
    )
    for uploaded_file in uploaded_files:
        st.text(f'Uploading {uploaded_file.name}...')
        file_content = StringIO(uploaded_file.getvalue().decode('utf-8')).read()
        file_sha1 = hashlib.sha1(uploaded_file.getbuffer()).hexdigest()
        vector_search.add_texts(
            texts=[
                file_content
            ],
            metadatas=[
                {
                    'file_name': uploaded_file.name,
                    'file_size': uploaded_file.size,
                    'embedding_model': embeddings_model,
                    'sha1': file_sha1,
                }
            ],
            ids=[f'{file_sha1}_{embeddings_model}']
        )
    st.write('Done!')
