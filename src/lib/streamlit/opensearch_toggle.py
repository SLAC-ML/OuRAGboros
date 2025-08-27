"""
Reusable OpenSearch toggle component with confirmation dialog
"""
import streamlit as st
import lib.config as config
import lib.streamlit.session_state as ss


def render_opensearch_toggle():
    """
    Renders the OpenSearch toggle with confirmation dialog.
    This is a reusable component for both main page and document_embedding page.
    """
    current_opensearch = st.session_state.get(ss.StateKey.USE_OPENSEARCH, config.prefer_opensearch)
    
    # Show current state with confirmation buttons
    storage_type = "OpenSearch (Persistent)" if current_opensearch else "In-Memory (Session only)"
    st.write(f"**Storage:** {storage_type}")
    
    # Show toggle button based on current state
    if current_opensearch:
        if st.button(
            "Switch to In-Memory Storage", 
            help="Switch to session-only storage (with confirmation)",
            use_container_width=True
        ):
            st.session_state["_pending_opensearch_change"] = False
            st.session_state["_show_opensearch_confirm"] = True
            st.rerun()
    else:
        if st.button(
            "Switch to OpenSearch Storage",
            help=f"Switch to persistent storage at {config.opensearch_base_url} (with confirmation)",
            use_container_width=True
        ):
            st.session_state["_pending_opensearch_change"] = True
            st.session_state["_show_opensearch_confirm"] = True
            st.rerun()


def render_opensearch_confirmation_dialog():
    """
    Renders the confirmation dialog when user wants to switch storage modes.
    Should be called outside the sidebar to avoid layout issues.
    """
    if st.session_state.get("_show_opensearch_confirm", False):
        @st.dialog("Change Storage Mode")
        def confirm_opensearch_change():
            pending_change = st.session_state.get("_pending_opensearch_change", False)
            
            if pending_change:
                st.write("**Switch to OpenSearch (Persistent Storage)**")
                st.info("""**What this means:**

- Knowledge bases will be stored in OpenSearch and persist across restarts
- You'll see your existing OpenSearch knowledge bases  
- Any current in-memory knowledge bases will become temporarily hidden
- You can switch back anytime to access in-memory knowledge bases again""")
            else:
                st.write("**Switch to In-Memory (Session Storage)**")
                st.info("""**What this means:**

- Knowledge bases will be stored in memory for this session only
- You'll see your current in-memory knowledge bases
- Your OpenSearch knowledge bases will become temporarily hidden
- You can switch back anytime to access OpenSearch knowledge bases again""")
            
            st.write("**Your data is always safe** - switching just changes which knowledge bases are currently accessible.")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Switch", key="confirm_opensearch_change", use_container_width=True):
                    # Actually apply the change
                    st.session_state[ss.StateKey.USE_OPENSEARCH] = pending_change
                    # Clear confirmation flags
                    del st.session_state["_pending_opensearch_change"]
                    del st.session_state["_show_opensearch_confirm"]
                    st.rerun()
            
            with col2:
                if st.button("Cancel", key="cancel_opensearch_change", use_container_width=True):
                    # Clear confirmation flags (no state change)
                    del st.session_state["_pending_opensearch_change"] 
                    del st.session_state["_show_opensearch_confirm"]
                    st.rerun()
        
        confirm_opensearch_change()