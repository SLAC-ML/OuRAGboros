import streamlit as st
import lib.nav as nav

st.set_page_config(
    page_title='Document Uploader',
    page_icon=':page_facing_up:',
    layout='wide',
)
nav.pages()

st.title(':page_facing_up: LangChain OuRAGborous')
