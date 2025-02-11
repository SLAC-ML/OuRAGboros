import streamlit as st

def pages():
    with st.sidebar:
        st.page_link('main.py', label='Home')
        st.page_link('pages/document_embedding.py', label='Document Embedding')
        st.divider()
