import io
import time

import streamlit as st

import lib.config as config
import lib.streamlit.nav as nav
import lib.streamlit.session_state as ss
import lib.streamlit.storage_toggle as storage_toggle
import lib.streamlit.kb_utils as kb_utils
import lib.rag_service as service

# Streamlit page configuration
st.set_page_config(
    page_title="OuRAGboros",
    page_icon=":snake:",
    layout="wide",
)

# Initialize navigation bar and session state
nav.pages()
available_llms, available_embeddings, available_knowledge_bases = ss.init()

st.title(":snake: LangChain OuRAGboros")

# Sidebar: Configuration
with st.sidebar:
    # STORAGE MODE TOGGLE WITH CONFIRMATION
    storage_toggle.render_storage_toggle()

    # Knowledge Base Management Section
    with st.container(border=True):
        st.subheader("Knowledge Base")

        # Get knowledge base list using utility with fallback handling
        current_available_kbs, storage_info = kb_utils.get_current_knowledge_bases(available_knowledge_bases)


        # Knowledge Base Selection - Use natural Streamlit state binding
        is_in_create_mode = st.session_state.get("_kb_create_mode_main", False)

        if not is_in_create_mode:
            # Normal mode: Simple selectbox with natural binding
            current_kb = st.session_state.get(ss.StateKey.KNOWLEDGE_BASE, "default")
            
            # CRITICAL: Clean up session state BEFORE selectbox renders to prevent serialization errors
            if current_kb not in current_available_kbs:
                current_kb = current_available_kbs[0]
                # Always update session state immediately to prevent selectbox serialization error
                st.session_state[ss.StateKey.KNOWLEDGE_BASE] = current_kb
            
            selected_kb = st.selectbox(
                "Select knowledge base:",
                current_available_kbs,
                index=current_available_kbs.index(current_kb),  # current_kb is now guaranteed to be in list
                key=ss.StateKey.KNOWLEDGE_BASE,  # Direct binding to session state!
                help="Choose an existing knowledge base",
            )
            
            # Add create button below the selectbox
            if st.button("+ Create New Knowledge Base", use_container_width=True):
                st.session_state["_kb_create_mode_main"] = True
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
                    key="new_kb_name_main",
                    label_visibility="collapsed",
                )

                col1, col2 = st.columns([1, 1])
                with col1:
                    create_clicked = st.button(
                        "Create", key="create_kb_btn_main", use_container_width=True
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
                                import lib.langchain.opensearch as langchain_opensearch

                                current_embedding = st.session_state[
                                    ss.StateKey.EMBEDDING_MODEL
                                ]

                                if st.session_state[ss.StateKey.USE_QDRANT]:
                                    # Create Qdrant collection
                                    import lib.langchain.qdrant as langchain_qdrant
                                    langchain_qdrant.ensure_qdrant_collection(
                                        current_embedding, new_kb_name
                                    )
                                elif st.session_state[ss.StateKey.USE_OPENSEARCH]:
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
                                st.session_state["_kb_create_mode_main"] = False
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
                        "Cancel", key="cancel_kb_main", use_container_width=True
                    )

                if cancel_clicked:
                    # Exit create mode and return to previous state
                    st.session_state["_kb_create_mode_main"] = False
                    # Clear the text input
                    if "new_kb_name_main" in st.session_state:
                        del st.session_state["new_kb_name_main"]
                    st.rerun()
        
        # Show current KB info and delete option (if not default) - always visible
        if not is_in_create_mode:
            current_kb = st.session_state.get(ss.StateKey.KNOWLEDGE_BASE, "default")
            if current_kb != "default":
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.caption(f"Active: **{current_kb}**")
                with col2:
                    if st.button(
                        "Delete",
                        key=f"delete_{current_kb}",
                        help=f"Delete '{current_kb}'",
                        use_container_width=True,
                    ):
                        st.session_state[f"_confirm_delete_{current_kb}"] = True
                        st.rerun()
            else:
                st.caption("Active: **default** (contains your original documents)")

    # st.header("Search Configuration")
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
        help="Score is computed using cosine similarity plus 1 to ensure a non-negative score.",
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

# Chat input: user query and optional file uploads
search_query = st.chat_input(
    "Enter a search query",
    accept_file=True,
    file_type=config.text_file_types,
)

if search_query:
    if search_query.text:
        st.session_state[ss.StateKey.SEARCH_QUERY] = search_query.text
        st.session_state[ss.StateKey.USER_CONTEXT] = [
            (ctx_file.name, io.StringIO(ctx_file.getvalue().decode('utf-8')).read())
            for ctx_file in search_query.files
        ]

# Show success messages if any
if "_delete_success_message" in st.session_state:
    st.success(st.session_state["_delete_success_message"])
    del st.session_state["_delete_success_message"]

if "_create_success_message" in st.session_state:
    st.success(st.session_state["_create_success_message"])
    del st.session_state["_create_success_message"]

# Handle storage mode toggle confirmation dialog
storage_toggle.render_storage_confirmation_dialog()

# Handle knowledge base deletion confirmations outside sidebar to avoid freezing issues
# Check for any deletion confirmation flags and handle them
kb_to_delete = None
# Check all possible KB names from session state keys
for key in st.session_state.keys():
    if key.startswith("_confirm_delete_"):
        kb_name = key.replace("_confirm_delete_", "")
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
                key=f"modal_confirm_del_{kb_to_delete}",
                use_container_width=True,
            ):
                try:
                    if st.session_state[ss.StateKey.USE_QDRANT]:
                        import lib.langchain.qdrant as langchain_qdrant

                        current_embedding = st.session_state[
                            ss.StateKey.EMBEDDING_MODEL
                        ]
                        # Delete Qdrant collection
                        client = langchain_qdrant.get_qdrant_client()
                        collection_name = langchain_qdrant.get_collection_name(
                            current_embedding, kb_to_delete
                        )
                        try:
                            client.delete_collection(collection_name=collection_name)
                        except Exception:
                            pass  # Collection might not exist
                    elif st.session_state[ss.StateKey.USE_OPENSEARCH]:
                        import lib.langchain.opensearch as langchain_opensearch

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

                    if f"_confirm_delete_{kb_to_delete}" in st.session_state:
                        del st.session_state[f"_confirm_delete_{kb_to_delete}"]
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
                key=f"modal_cancel_del_{kb_to_delete}",
                use_container_width=True,
            ):
                if f"_confirm_delete_{kb_to_delete}" in st.session_state:
                    del st.session_state[f"_confirm_delete_{kb_to_delete}"]
                st.rerun()

    confirm_delete()

# Function to render retrieved source documents

def _render_source_docs(docs):
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
        st.code(
            f"{doc.page_content[:1000]}" +
            ('... [download file to see more]' if len(doc.page_content) > 100 else ''),
            language=None,
            line_numbers=True,
            wrap_lines=True,
        )

# Show storage switching notice if confirmation dialog is active
if st.session_state.get("_show_storage_confirm", False):
    st.info("🔄 Storage mode change pending - complete the dialog in the sidebar to continue.")

# Main chat loop: perform RAG and display results
# Skip RAG execution if storage switch is pending to avoid interruption
if (
    st.session_state.get(ss.StateKey.SEARCH_QUERY) and
    st.session_state.get(ss.StateKey.LLM_MODEL) and
    not st.session_state.get("_show_storage_confirm", False)
):
    # Echo user query
    with st.chat_message("user"):
        st.text(st.session_state[ss.StateKey.SEARCH_QUERY])

    # Call into our shared RAG service
    with st.spinner("Running retrieval + LLM…"):
        # Debug info for troubleshooting
        current_kb = st.session_state[ss.StateKey.KNOWLEDGE_BASE]
        st.caption(f"Querying knowledge base: **{current_kb}**")

        answer, docs = service.answer_query(
            query=st.session_state[ss.StateKey.SEARCH_QUERY],
            embedding_model=st.session_state[ss.StateKey.EMBEDDING_MODEL],
            llm_model=st.session_state[ss.StateKey.LLM_MODEL],
            k=st.session_state[ss.StateKey.MAX_DOCUMENT_COUNT],
            score_threshold=st.session_state[ss.StateKey.QUERY_RESULT_SCORE_INF],
            use_opensearch=st.session_state[ss.StateKey.USE_OPENSEARCH],
            prompt_template=st.session_state[ss.StateKey.LLM_PROMPT],
            user_files=st.session_state.get(ss.StateKey.USER_CONTEXT, []),
            knowledge_base=current_kb,
            use_qdrant=st.session_state[ss.StateKey.USE_QDRANT],
        )

    # Display the LLM’s answer
    with st.chat_message("ai"):
        #st.text(answer)
        st.markdown(answer)

    # Store & render source docs
    st.session_state[ss.StateKey.RAG_DOCS] = docs
    if docs:
        singular = len(docs) == 1
        with st.expander(f"Source document{'s' if not singular else ''}"):
            _render_source_docs(docs)

    # Allow user to download session
    st.download_button(
        "Download session data",
        ss.dump_session_state(),
        file_name=f"ouragboros_session_data_{time.time()}.json",
        mime="application/json",
    )
