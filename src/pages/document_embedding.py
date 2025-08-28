import hashlib
import io
import pathlib
import time

import streamlit as st
from langchain_core.vectorstores import VectorStore

import lib.streamlit.nav as nav
import lib.streamlit.session_state as ss
import lib.streamlit.opensearch_toggle as opensearch_toggle
import lib.streamlit.kb_utils as kb_utils

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
    # OPENSEARCH TOGGLE WITH CONFIRMATION
    opensearch_toggle.render_opensearch_toggle()

    # Knowledge Base Management Section (same simplified UI as main page)
    with st.container(border=True):
        st.subheader("Knowledge Base")

        # Get knowledge base list using utility with fallback handling
        current_available_kbs, storage_info = kb_utils.get_current_knowledge_bases(available_knowledge_bases)

        # Knowledge Base Selection - Use natural Streamlit state binding (same as main page)
        is_in_create_mode = st.session_state.get("_kb_create_mode_embed", False)

        if not is_in_create_mode:
            # Normal mode: Simple selectbox with natural binding
            current_kb = st.session_state.get(ss.StateKey.KNOWLEDGE_BASE, "default")
            
            # CRITICAL: Clean up session state BEFORE selectbox renders to prevent serialization errors
            if current_kb not in current_available_kbs:
                current_kb = current_available_kbs[0]
                # Always update session state immediately to prevent selectbox serialization error
                st.session_state[ss.StateKey.KNOWLEDGE_BASE] = current_kb
            
            selected_kb = st.selectbox(
                "Target knowledge base for uploads:",
                current_available_kbs,
                index=current_available_kbs.index(current_kb),  # current_kb is now guaranteed to be in list
                key=ss.StateKey.KNOWLEDGE_BASE,  # Direct binding to session state!
                help="Choose where to store uploaded documents",
            )
            
            # Add create button below the selectbox
            if st.button("+ Create New Knowledge Base", use_container_width=True, key="create_kb_btn_embed_main"):
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
                        "Create",
                        key="create_kb_btn_embed",
                        use_container_width=True,
                    )

                if create_clicked:
                    if not new_kb_name or new_kb_name.strip() == "":
                        st.error(
                            "💡 Please enter a name for your knowledge base. Only letters, numbers, and underscores are allowed."
                        )
                    elif new_kb_name in current_available_kbs:
                        st.error(
                            f"💡 Knowledge base '{new_kb_name}' already exists! Please choose a different name."
                        )
                    else:
                        import re

                        if re.match(r"^[a-zA-Z0-9_]+$", new_kb_name):
                            try:
                                current_embedding = st.session_state[
                                    ss.StateKey.EMBEDDING_MODEL
                                ]

                                if st.session_state[ss.StateKey.USE_OPENSEARCH]:
                                    langchain_opensearch.ensure_opensearch_index(
                                        current_embedding, new_kb_name
                                    )
                                else:
                                    if (
                                        "_in_memory_knowledge_bases"
                                        not in st.session_state
                                    ):
                                        st.session_state[
                                            "_in_memory_knowledge_bases"
                                        ] = []
                                    st.session_state[
                                        "_in_memory_knowledge_bases"
                                    ].append(new_kb_name)

                                # Auto-select the new KB and refresh
                                st.session_state[ss.StateKey.KNOWLEDGE_BASE] = new_kb_name
                                st.cache_resource.clear()
                                # Exit create mode and clean up
                                st.session_state["_kb_create_mode_embed"] = False
                                # Set success message flag to show outside section
                                st.session_state["_create_success_message"] = (
                                    f"Created '{new_kb_name}' successfully!"
                                )
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
                    st.rerun()
        
        # Show current KB info and delete option (if not default) - always visible
        if not is_in_create_mode:
            current_kb = st.session_state.get(ss.StateKey.KNOWLEDGE_BASE, "default")
            if current_kb != "default":
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.caption(f"Upload target: **{current_kb}**")
                with col2:
                    if st.button(
                        "Delete",
                        key=f"delete_embed_{current_kb}",
                        help=f"Delete '{current_kb}'",
                        use_container_width=True,
                    ):
                        st.session_state[f"_confirm_delete_embed_{current_kb}"] = True
                        st.rerun()
            else:
                st.caption("Upload target: **default** (your original knowledge base)")

    # st.header('Document Embedding Configuration')
    # st.text("Upload and embed documents into the selected knowledge base.")

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
st.info(
    f"**Current Configuration:**\n"
    f"- Knowledge Base: `{current_kb}`\n"
    f"- Embedding Model: `{st.session_state[ss.StateKey.EMBEDDING_MODEL]}`\n\n"
    f"Documents will be embedded and stored in the selected knowledge base using the selected model. "
    f"Different knowledge bases are completely isolated from each other."
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

# Show success messages if any
if "_delete_success_message" in st.session_state:
    st.success(st.session_state["_delete_success_message"])
    del st.session_state["_delete_success_message"]

if "_create_success_message" in st.session_state:
    st.success(st.session_state["_create_success_message"])
    del st.session_state["_create_success_message"]

# Handle knowledge base deletion confirmations outside sidebar to avoid freezing issues
# Check for any deletion confirmation flags and handle them
kb_to_delete = None
# Use session state keys to find deletion flags (same pattern as main.py)
for key in st.session_state.keys():
    if key.startswith("_confirm_delete_embed_"):
        kb_name = key.replace("_confirm_delete_embed_", "")
        if kb_name != "default" and st.session_state.get(key, False):
            kb_to_delete = kb_name
            break

# Handle deletion confirmation dialog for the selected knowledge base
if kb_to_delete:

    @st.dialog(f"Delete Knowledge Base: {kb_to_delete}")
    def confirm_delete():
        st.write(
            f"Are you sure you want to delete **{kb_to_delete}** and all its documents?"
        )
        st.write("This action cannot be undone.")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "Yes, delete",
                key=f"modal_confirm_del_embed_{kb_to_delete}",
                use_container_width=True,
            ):
                try:
                    if st.session_state[ss.StateKey.USE_OPENSEARCH]:
                        current_embedding = st.session_state[
                            ss.StateKey.EMBEDDING_MODEL
                        ]
                        langchain_opensearch.delete_knowledge_base(
                            kb_to_delete, current_embedding
                        )
                    else:
                        if "_in_memory_knowledge_bases" in st.session_state:
                            if (
                                kb_to_delete
                                in st.session_state["_in_memory_knowledge_bases"]
                            ):
                                st.session_state["_in_memory_knowledge_bases"].remove(
                                    kb_to_delete
                                )

                    # Switch to default and refresh
                    st.session_state[ss.StateKey.KNOWLEDGE_BASE] = "default"
                    st.cache_resource.clear()
                    if f"_confirm_delete_embed_{kb_to_delete}" in st.session_state:
                        del st.session_state[f"_confirm_delete_embed_{kb_to_delete}"]
                    # Set success message flag to show outside dialog
                    st.session_state["_delete_success_message"] = (
                        f"Deleted '{kb_to_delete}' successfully!"
                    )
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to delete: {str(e)}")

        with col2:
            if st.button(
                "Cancel",
                key=f"modal_cancel_del_embed_{kb_to_delete}",
                use_container_width=True,
            ):
                if f"_confirm_delete_embed_{kb_to_delete}" in st.session_state:
                    del st.session_state[f"_confirm_delete_embed_{kb_to_delete}"]
                st.rerun()

    confirm_delete()

# Handle OpenSearch toggle confirmation dialog
opensearch_toggle.render_opensearch_confirmation_dialog()
