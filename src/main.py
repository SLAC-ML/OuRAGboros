import io
import time

import streamlit as st

import lib.config as config
import lib.streamlit.nav as nav
import lib.streamlit.session_state as ss
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
    st.toggle(
        "Use OpenSearch",
        key=ss.StateKey.USE_OPENSEARCH,
        help=f"Requires an OpenSearch instance running at {config.opensearch_base_url}. "
             "If this toggle is off, all documents are retrieved from an in-memory vector "
             "store which is lost when the application terminates.",
    )

    # Knowledge Base Management Section
    with st.container(border=True):
        st.subheader("Knowledge Base")

        # Check if we're in create mode
        is_in_create_mode = st.session_state.get("_kb_create_mode_main", False)
        
        # Determine what to display in the selectbox
        if is_in_create_mode:
            display_selection = st.session_state.get("_original_kb_main", "default")
        else:
            display_selection = st.session_state.get(ss.StateKey.KNOWLEDGE_BASE, "default")

        # Enhanced options list: existing KBs + "Create new..."
        kb_options = available_knowledge_bases + ["+ Create new..."]

        # Knowledge Base Selection
        selected_option = st.selectbox(
            "Select or create knowledge base:",
            kb_options,
            index=kb_options.index(display_selection) if display_selection in available_knowledge_bases else 0,
            key="kb_selector_main",
            help="Choose an existing knowledge base or create a new one",
            disabled=is_in_create_mode,
        )

        # Handle "Create new" selection
        # Check if we should ignore this selection (e.g., after cancel)
        ignore_create_selection = st.session_state.get("_ignore_create_selection_main", False)
        if ignore_create_selection:
            # Clear the ignore flag and don't enter create mode
            del st.session_state["_ignore_create_selection_main"]
        elif selected_option == "+ Create new..." and not is_in_create_mode:
            # Store current KB selection before entering create mode
            st.session_state["_original_kb_main"] = st.session_state.get(ss.StateKey.KNOWLEDGE_BASE, "default")
            # Set create mode state
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
                    
                    if create_clicked and new_kb_name:
                        import re

                        if re.match(r"^[a-zA-Z0-9_]+$", new_kb_name):
                            if new_kb_name in available_knowledge_bases:
                                st.error(f"'{new_kb_name}' already exists!")
                            else:
                                try:
                                    import lib.langchain.opensearch as langchain_opensearch

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
                                    st.session_state["_kb_create_mode_main"] = False
                                    if "new_kb_name_main" in st.session_state:
                                        del st.session_state["new_kb_name_main"]
                                    if "_original_kb_main" in st.session_state:
                                        del st.session_state["_original_kb_main"]
                                    st.session_state[ss.StateKey.KNOWLEDGE_BASE] = new_kb_name
                                    st.cache_resource.clear()
                                    
                                    # Set success message for main content area
                                    with st.container():
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
                        "Cancel", key="cancel_kb_main", use_container_width=True
                    )
                    
                if cancel_clicked:
                    # Exit create mode and return to previous state
                    st.session_state["_kb_create_mode_main"] = False
                    # Clear the text input
                    if "new_kb_name_main" in st.session_state:
                        del st.session_state["new_kb_name_main"]
                    # Clear the original KB storage
                    if "_original_kb_main" in st.session_state:
                        del st.session_state["_original_kb_main"]
                    # Set a flag to ignore the "+ Create new..." selection on next run
                    st.session_state["_ignore_create_selection_main"] = True
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
                st.caption(f"Active: **{selected_option}**")
            else:
                st.caption("Active: **default** (contains your original documents)")

    st.header("Search Configuration")
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

# Main chat loop: perform RAG and display results
if (
    st.session_state.get(ss.StateKey.SEARCH_QUERY) and
    st.session_state.get(ss.StateKey.LLM_MODEL)
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
