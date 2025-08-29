"""
Storage mode toggle component supporting In-Memory, OpenSearch, and Qdrant
"""
import streamlit as st
import lib.config as config
import lib.streamlit.session_state as ss


def get_storage_modes():
    """Get available storage modes based on configuration"""
    modes = ["In-Memory"]
    
    # OpenSearch is always available
    modes.append("OpenSearch")
    
    # Qdrant is available if configured
    if hasattr(config, 'prefer_qdrant') and getattr(config, 'qdrant_base_url', None):
        modes.append("Qdrant")
    
    return modes


def get_current_storage_mode():
    """Get the current storage mode from session state"""
    if st.session_state.get(ss.StateKey.USE_QDRANT, getattr(config, 'prefer_qdrant', False)):
        return "Qdrant"
    elif st.session_state.get(ss.StateKey.USE_OPENSEARCH, config.prefer_opensearch):
        return "OpenSearch"
    else:
        return "In-Memory"


def render_storage_toggle():
    """
    Renders the storage mode selector with confirmation dialog.
    """
    available_modes = get_storage_modes()
    current_mode = get_current_storage_mode()
    
    # Storage mode descriptions
    mode_descriptions = {
        "In-Memory": "Session only",
        "OpenSearch": "Persistent (Legacy)",
        "Qdrant": "High-Performance"
    }
    
    # Show current state
    desc = mode_descriptions.get(current_mode, "")
    st.write(f"**Vector Storage:** {current_mode}")
    if desc:
        st.caption(desc)
    
    # Storage mode selector
    if len(available_modes) > 1:
        st.write("**Switch to:**")
        
        for mode in available_modes:
            if mode != current_mode:
                desc = mode_descriptions.get(mode, "")
                button_text = f"{mode}" + (f" ({desc})" if desc else "")
                
                if st.button(
                    button_text,
                    use_container_width=True,
                    key=f"switch_to_{mode.lower()}"
                ):
                    st.session_state["_pending_storage_change"] = mode
                    st.session_state["_show_storage_confirm"] = True


def render_storage_confirmation_dialog():
    """
    Renders the confirmation dialog when user wants to switch storage modes.
    Should be called outside the sidebar to avoid layout issues.
    """
    if st.session_state.get("_show_storage_confirm", False):
        @st.dialog("Change Vector Storage Mode")
        def confirm_storage_change():
            pending_mode = st.session_state.get("_pending_storage_change", "In-Memory")
            current_mode = get_current_storage_mode()
            
            st.write(f"**Switch from {current_mode} to {pending_mode}**")
            
            # Mode-specific information
            if pending_mode == "Qdrant":
                st.info("""**🚀 Qdrant (High-Performance Mode)**

**Benefits:**
- ⚡ 100x faster searches (0.03s vs 60s+ response times)
- 🎯 Handles 100+ concurrent users easily  
- 🔄 360+ requests/second throughput
- 💾 Persistent storage across restarts

**What happens:**
- Knowledge bases stored in high-performance Qdrant vector database
- You'll see your existing Qdrant collections (if any)
- Other storage modes become temporarily hidden
- Best choice for production workloads""")
                
            elif pending_mode == "OpenSearch":
                st.info("""**📚 OpenSearch (Legacy Persistent Storage)**

**Features:**
- 💾 Persistent storage across restarts
- 🔍 Full-text search capabilities
- 📊 Good for small to medium workloads

**Performance Note:**
- ⚠️ Slower for high concurrency (60s+ response times)
- 🐌 Not recommended for 100+ concurrent users

**What happens:**
- Knowledge bases stored in OpenSearch indices
- You'll see your existing OpenSearch knowledge bases
- Other storage modes become temporarily hidden""")
                
            else:  # In-Memory
                st.info("""**💨 In-Memory (Session Storage)**

**Features:**
- ⚡ Fast for development and testing
- 🧪 No setup required
- 🔄 Data cleared on restart

**What happens:**
- Knowledge bases stored in memory for this session only
- You'll see your current in-memory knowledge bases
- Persistent storage becomes temporarily hidden
- Good for quick testing and development""")

            st.write("**Your data is always safe** - switching just changes which knowledge bases are currently accessible.")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Switch", key="confirm_storage_change", use_container_width=True):
                    # Apply the storage change
                    if pending_mode == "Qdrant":
                        st.session_state[ss.StateKey.USE_QDRANT] = True
                        st.session_state[ss.StateKey.USE_OPENSEARCH] = False
                    elif pending_mode == "OpenSearch":
                        st.session_state[ss.StateKey.USE_QDRANT] = False  
                        st.session_state[ss.StateKey.USE_OPENSEARCH] = True
                    else:  # In-Memory
                        st.session_state[ss.StateKey.USE_QDRANT] = False
                        st.session_state[ss.StateKey.USE_OPENSEARCH] = False
                    
                    # Clear confirmation flags
                    del st.session_state["_pending_storage_change"]
                    del st.session_state["_show_storage_confirm"]
                    
                    # Clear cached resources to refresh knowledge bases
                    st.cache_resource.clear()
                    st.rerun()

            with col2:
                if st.button("Cancel", key="cancel_storage_change", use_container_width=True):
                    # Clear confirmation flags (no state change)
                    del st.session_state["_pending_storage_change"]
                    del st.session_state["_show_storage_confirm"]
                    st.rerun()

        confirm_storage_change()