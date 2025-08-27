"""
Knowledge base utilities for consistent KB list management across pages
"""
import streamlit as st
import lib.streamlit.session_state as ss
import lib.langchain.opensearch as langchain_opensearch


def get_current_knowledge_bases(available_knowledge_bases_fallback):
    """
    Get current knowledge base list based on OpenSearch toggle state.
    
    This function provides consistent KB list logic across all pages with proper
    fallback handling when OpenSearch is not ready on first page load.
    
    Args:
        available_knowledge_bases_fallback: Cached KB list from ss.init() as fallback
        
    Returns:
        tuple: (current_available_kbs, storage_info)
    """
    current_use_opensearch = st.session_state.get(
        ss.StateKey.USE_OPENSEARCH, 
        st.session_state.get('prefer_opensearch', True)
    )
    
    if current_use_opensearch:
        try:
            current_available_kbs = langchain_opensearch.get_available_knowledge_bases()
            storage_info = "📊 OpenSearch (Persistent)"
        except Exception as e:
            # First, try a quick retry (OpenSearch might just be slow to respond)
            try:
                import time
                time.sleep(0.1)  # Brief pause
                current_available_kbs = langchain_opensearch.get_available_knowledge_bases()
                storage_info = "📊 OpenSearch (Persistent)"
            except Exception:
                # Fallback to cached list from ss.init() if OpenSearch not ready
                opensearch_kbs = [kb for kb in available_knowledge_bases_fallback if kb != "default"] 
                current_available_kbs = ["default"] + opensearch_kbs
                storage_info = "⚠️ OpenSearch initializing..."
    else:
        # In-memory mode: only show in-memory KBs
        current_available_kbs = ["default"]
        if "_in_memory_knowledge_bases" in st.session_state:
            current_available_kbs.extend([
                kb for kb in st.session_state["_in_memory_knowledge_bases"]
                if kb not in current_available_kbs
            ])
        storage_info = "💾 In-Memory (Session only)"
    
    return current_available_kbs, storage_info