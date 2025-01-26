import os
from pathlib import Path

import pandas as pd
import streamlit as st

import lib.nav as nav
import lib.config as config

st.set_page_config(
    page_title='Document Uploader',
    page_icon=':page_facing_up:',
    layout='wide',
)
nav.pages()

st.title(':page_facing_up: Document Upload')

desired_file_extension = '.md'
document_search_glob = f'*{desired_file_extension}'

# Initialize session state
#
if 'root_doc_path' not in st.session_state:
    st.session_state.root_doc_path = config.default_root_doc_path

root_doc_path = st.text_input(
    label='Root document path:',
    help='This folder that will be recursively indexed for search.',
    key='root_doc_path'
)
if root_doc_path != st.session_state.root_doc_path:
    st.session_state.root_doc_path = root_doc_path

if not os.path.exists(st.session_state.root_doc_path):
    raise ValueError('Selected root document path does not exist.')

docs = list(Path(st.session_state.root_doc_path).rglob(document_search_glob))
doc_df = pd.DataFrame([
    (f'.{str(doc).removeprefix(st.session_state.root_doc_path)}',
     doc.stat().st_size
     ) for doc in docs
], columns=['Name', 'Size [bytes]'])

st.text('Found the following documents:')
st.dataframe(doc_df, use_container_width=True)

