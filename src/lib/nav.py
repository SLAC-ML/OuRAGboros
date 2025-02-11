import streamlit as st

def pages():
    with st.sidebar:
        st.page_link('main.py', label='Home')
        st.page_link('pages/document_uploader.py', label='Document Upload')
        st.divider()
