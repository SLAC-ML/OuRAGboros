import hashlib
import io
import pathlib
import time

import streamlit as st
from langchain_core.vectorstores import VectorStore

import lib.streamlit.nav as nav
import lib.streamlit.session_state as ss

import lib.langchain.embeddings as langchain_embeddings
import lib.langchain.models as langchain_models
import lib.langchain.opensearch as langchain_opensearch

import lib.config as config

st.set_page_config(
    page_title='Document Embedding',
    page_icon=':page_facing_up:',
    layout='wide',
)

# Initialize navigation bar
#
nav.pages()

# Initialize session state
#
available_llms, available_embeddings = ss.init()

st.title(':page_facing_up: Document Embedding')


def _sliding_window(sequence: iter, window_size: int, step_size: int):
    """
    Collect data into overlapping fixed-length chunks or blocks.

    _sliding_window('ABCDEFG', 4, 2) → ['ABCD','CDEF','EFG']
    """
    if 0 < window_size < len(sequence) and 0 < step_size:
        for i in range(0, len(sequence) - step_size + 1, step_size):
            yield sequence[i: i + window_size]
    else:
        yield sequence


def _upload_text_to_vector_store(
        vs: VectorStore,
        embedding_model_name: str,
        text_file_bytes: io.BytesIO,
        text_file_name: str,
        text_chunk_size: int = -1,
        text_chunk_overlap: int = 0,
        text_page: int = 1,
):
    text_bytes = text_file_bytes.getvalue()
    file_content = text_bytes.decode('utf-8')
    file_hash = hashlib.sha256(text_file_bytes.getbuffer()).hexdigest()

    overlap = int((100 - text_chunk_overlap) * text_chunk_size / 100)
    chunks = list(_sliding_window(
        file_content,
        text_chunk_size,
        overlap,
    ))

    timestamp = int(time.time())
    for i, chunk in enumerate(chunks):
        opensearch_document_id = hashlib.sha256(
            '{}_{}_{}_{}_{}_{}'.format(
                file_hash,
                embedding_model_name,
                text_page,
                text_chunk_size,
                text_chunk_overlap,
                i,
            ).encode('utf-8')).hexdigest()
        metadata = {
            'embedding_model': embedding_model_name,
            'source': text_file_name,
            'page_number': text_page,
            'chunk_size': len(chunk),
            'chunk_overlap_percent': text_chunk_overlap,
            'chunk_index': i,
            'chunks': len(chunks),
            'uploaded': timestamp,
        }

        yield (i, vs.add_texts(
            texts=[chunk],
            ids=[opensearch_document_id],
            metadatas=[metadata],
        ), len(chunks))


with st.sidebar:
    st.header('Search Configuration')
    st.toggle(
        "Use OpenSearch",
        key=ss.StateKey.USE_OPENSEARCH,
        help=f"Requires an OpenSearch instance running at {config.opensearch_base_url}. "
             "If this toggle is off, all documents are stored in an in-memory vector "
             "store which is lost when the application terminates.",
    )
    st.selectbox(
        "Embedding model:",
        available_embeddings,
        key=ss.StateKey.EMBEDDING_MODEL
    )
    st.number_input(
        'Text chunk size [characters]:',
        min_value=0,
        step=50,
        key=ss.StateKey.EMBEDDING_CHUNK_SIZE,
        help='Used to split each text file (or PDF page) into smaller chunks for '
             'embedding. A value of `0` will embed each page (or text document) in its '
             'entirety.'
    )
    if st.session_state[ss.StateKey.EMBEDDING_CHUNK_SIZE] > 0:
        st.slider(
            'Text chunk overlap [%]:',
            min_value=0,
            max_value=50,
            step=5,
            key=ss.StateKey.EMBEDDING_CHUNK_OVERLAP,
            help='Specifies the percent overlap allowed between text chunks.',
        )

st.warning(
    'IMPORTANT: When you upload documents, they are embedded _only_ for the '
    'selected embedding model (currently '
    f'`{st.session_state[ss.StateKey.EMBEDDING_MODEL]}`). When uploading documents,'
    f' please be sure that you embed your source texts with every model you wish to use.'
)

# Upload text documents to OpenSearch
#
uploaded_files = st.file_uploader(
    'Upload files to be added to the application knowledgebase.',
    accept_multiple_files=True,
    type=[*config.text_file_types, 'pdf'],
)
if len(uploaded_files) and st.button('Embed Text'):
    # Create OpenSearch index if it doesn't already exist
    #
    with st.spinner('Loading `{}` embeddings...'.format(
            st.session_state[ss.StateKey.EMBEDDING_MODEL]
    )):
        langchain_models.pull_model(st.session_state[ss.StateKey.EMBEDDING_MODEL])

    if st.session_state[ss.StateKey.USE_OPENSEARCH]:
        st.text('Ensuring OpenSearch index existence...')
        langchain_opensearch.ensure_opensearch_index(
            st.session_state[ss.StateKey.EMBEDDING_MODEL]
        )
        vector_store = langchain_opensearch.opensearch_doc_vector_store(
            st.session_state[ss.StateKey.EMBEDDING_MODEL]
        )
    else:
        vector_store = langchain_embeddings.get_in_memory_vector_store(
            st.session_state[ss.StateKey.EMBEDDING_MODEL]
        )

    text_upload_progress = st.progress(0)

    for k, uploaded_file in enumerate(uploaded_files):
        text_upload_progress.progress(
            (k + 1) / len(uploaded_files),
            f'Embedding {uploaded_file.name} [{k + 1}/{len(uploaded_files)}]...'
        )

        if pathlib.Path(uploaded_file.name).suffix.lower() == '.pdf':
            # Lazy import since the imports this class brings in are a little hefty.
            #
            from lib.pdf.nougat_extractor import NougatExtractor

            pdf_extractor = NougatExtractor()
            pdf_progress = st.progress(0)

            for j, (txt_bytes, txt_name, pages) in enumerate(
                    pdf_extractor.extract_text(uploaded_file, uploaded_file.name)
            ):
                pdf_progress.progress(
                    (j + 1) / len(pages),
                    f'Processing {txt_name} page [{j + 1}' f'/{len(pages)}]...'
                )

                chunk_progress = st.progress(0)
                for chunk_index, _, chunk_count in _upload_text_to_vector_store(
                        vs=vector_store,
                        embedding_model_name=st.session_state[
                            ss.StateKey.EMBEDDING_MODEL
                        ],
                        text_file_bytes=txt_bytes,
                        text_file_name=uploaded_file.name,
                        text_chunk_size=st.session_state[
                            ss.StateKey.EMBEDDING_CHUNK_SIZE
                        ],
                        text_chunk_overlap=st.session_state[
                            ss.StateKey.EMBEDDING_CHUNK_OVERLAP
                        ],
                        text_page=j + 1,
                ):
                    chunk_progress.progress(
                        (chunk_index + 1) / chunk_count,
                        f'Embedding chunk [{chunk_index + 1}' f'/{chunk_count}]...'
                    )

                chunk_progress.empty()
        else:
            # Process raw text
            #
            chunk_progress = st.progress(0)
            for chunk_index, _, chunk_count in _upload_text_to_vector_store(
                    vs=vector_store,
                    embedding_model_name=st.session_state[
                        ss.StateKey.EMBEDDING_MODEL
                    ],
                    text_file_bytes=uploaded_file,
                    text_file_name=uploaded_file.name,
                    text_chunk_size=st.session_state[
                        ss.StateKey.EMBEDDING_CHUNK_SIZE
                    ],
                    text_chunk_overlap=st.session_state[
                        ss.StateKey.EMBEDDING_CHUNK_OVERLAP
                    ],
                    text_page=1,
            ):
                chunk_progress.progress(
                    (chunk_index + 1) / chunk_count,
                    f'Embedding chunk [{chunk_index + 1}' f'/{chunk_count}]...'
                )

            chunk_progress.empty()

    st.write('Done!')
