import hashlib
import io
import pathlib
import time

import streamlit as st
from langchain_core.vectorstores import VectorStore

import lib.streamlit.nav as nav
import lib.streamlit.session_state as ss

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
available_llms, available_embeddings, available_knowledge_bases = ss.init()

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
    st.toggle(
        "Use OpenSearch",
        key=ss.StateKey.USE_OPENSEARCH,
        help=f"Requires an OpenSearch instance running at {config.opensearch_base_url}. "
             "If this toggle is off, all documents are stored in an in-memory vector "
             "store which is lost when the application terminates.",
    )

    # Knowledge Base Management Section (same as main page)
    with st.container(border=True):
        st.subheader("Knowledge Base")

        # Check if we're in create mode
        is_in_create_mode = st.session_state.get("_kb_create_mode_embed", False)
        
        # Determine what to display in the selectbox
        if is_in_create_mode:
            display_selection = st.session_state.get("_original_kb_embed", "default")
        else:
            display_selection = st.session_state.get(ss.StateKey.KNOWLEDGE_BASE, "default")

        # Enhanced options list: existing KBs + "Create new..."
        kb_options = available_knowledge_bases + ["+ Create new..."]

        # Knowledge Base Selection
        selected_option = st.selectbox(
            "Target knowledge base for uploads:",
            kb_options,
            index=kb_options.index(display_selection) if display_selection in available_knowledge_bases else 0,
            key="kb_selector_embed",
            help="Choose where to store uploaded documents",
            disabled=is_in_create_mode,
        )

        # Handle "Create new" selection
        # Check if we should ignore this selection (e.g., after cancel)
        ignore_create_selection = st.session_state.get("_ignore_create_selection_embed", False)
        if ignore_create_selection:
            # Clear the ignore flag and don't enter create mode
            del st.session_state["_ignore_create_selection_embed"]
        elif selected_option == "+ Create new..." and not is_in_create_mode:
            # Store current KB selection before entering create mode
            st.session_state["_original_kb_embed"] = st.session_state.get(ss.StateKey.KNOWLEDGE_BASE, "default")
            # Set create mode state
            st.session_state["_kb_create_mode_embed"] = True
            st.rerun()
        
        # Show create mode UI if in create mode
        if is_in_create_mode:
            with st.container():
                st.write("**Create New Knowledge Base**")
                st.info("💡 Enter a name for your new knowledge base below:")
                new_kb_name = st.text_input(
                    "Name:",
                    placeholder="e.g., physics_papers, legal_docs",
                    help="Only letters, numbers, and underscores allowed",
                    key="new_kb_name_embed",
                    label_visibility="collapsed",
                )

                col1, col2 = st.columns([1, 1])
                with col1:
                    create_clicked = st.button(
                        "Create", key="create_kb_btn_embed", use_container_width=True
                    )
                    
                    if create_clicked and new_kb_name:
                        import re

                        if re.match(r"^[a-zA-Z0-9_]+$", new_kb_name):
                            if new_kb_name in available_knowledge_bases:
                                st.error(f"'{new_kb_name}' already exists!")
                            else:
                                try:
                                    current_embedding = st.session_state[
                                        ss.StateKey.EMBEDDING_MODEL
                                    ]
                                    
                                    # Create the knowledge base by ensuring the index exists
                                    if st.session_state[ss.StateKey.USE_OPENSEARCH]:
                                        langchain_opensearch.ensure_opensearch_index(
                                            current_embedding, new_kb_name
                                        )
                                    else:
                                        # For in-memory, add to tracking list
                                        if "_in_memory_knowledge_bases" not in st.session_state:
                                            st.session_state["_in_memory_knowledge_bases"] = []
                                        if new_kb_name not in st.session_state["_in_memory_knowledge_bases"]:
                                            st.session_state["_in_memory_knowledge_bases"].append(new_kb_name)
                                    
                                    # Exit create mode and set new KB
                                    st.session_state["_kb_create_mode_embed"] = False
                                    if "new_kb_name_embed" in st.session_state:
                                        del st.session_state["new_kb_name_embed"]
                                    if "_original_kb_embed" in st.session_state:
                                        del st.session_state["_original_kb_embed"]
                                    st.session_state[ss.StateKey.KNOWLEDGE_BASE] = new_kb_name
                                    st.cache_resource.clear()
                                    st.success(f"Knowledge base '{new_kb_name}' created successfully!")
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Failed to create: {str(e)}")
                        else:
                            st.error(
                                "💡 Invalid name! Knowledge base names can only contain letters, numbers, and underscores (e.g., 'physics_papers', 'legal_docs')"
                            )

                with col2:
                    cancel_clicked = st.button(
                        "Cancel", key="cancel_kb_embed", use_container_width=True
                    )
                    
                if cancel_clicked:
                    # Exit create mode and return to previous state
                    st.session_state["_kb_create_mode_embed"] = False
                    # Clear the text input
                    if "new_kb_name_embed" in st.session_state:
                        del st.session_state["new_kb_name_embed"]
                    # Clear the original KB storage
                    if "_original_kb_embed" in st.session_state:
                        del st.session_state["_original_kb_embed"]
                    # Set a flag to ignore the "+ Create new..." selection on next run
                    st.session_state["_ignore_create_selection_embed"] = True
                    st.rerun()

        else:
            # Update the actual knowledge base selection (but ignore "+ Create new...")
            if (selected_option != st.session_state.get(ss.StateKey.KNOWLEDGE_BASE) and 
                selected_option != "+ Create new..." and 
                selected_option in available_knowledge_bases):
                st.session_state[ss.StateKey.KNOWLEDGE_BASE] = selected_option
                st.cache_resource.clear()  # Clear cache when switching KBs
                st.rerun()

            # Show current KB info
            if selected_option != "default":
                st.caption(f"Target: **{selected_option}**")
            else:
                st.caption("Target: **default** (your original knowledge base)")

    st.header('Document Embedding Configuration')
    st.text("Upload and embed documents into the selected knowledge base.")
    
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

current_kb = st.session_state[ss.StateKey.KNOWLEDGE_BASE]
st.warning(
    f'IMPORTANT: Documents will be uploaded to the **{current_kb}** knowledge base '
    f'and embedded with the **{st.session_state[ss.StateKey.EMBEDDING_MODEL]}** model. '
    'Documents are isolated per knowledge base and embedding model combination.'
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

    vector_store = ss.get_vector_store(
        st.session_state[ss.StateKey.USE_OPENSEARCH],
        st.session_state[ss.StateKey.EMBEDDING_MODEL],
        st.session_state[ss.StateKey.KNOWLEDGE_BASE]
    )
    if st.session_state[ss.StateKey.USE_OPENSEARCH]:
        st.text('Ensuring OpenSearch index existence...')
        langchain_opensearch.ensure_opensearch_index(
            st.session_state[ss.StateKey.EMBEDDING_MODEL],
            st.session_state[ss.StateKey.KNOWLEDGE_BASE]
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
