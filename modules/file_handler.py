# modules/file_handler.py

import streamlit as st
from langchain.schema import Document

def handle_file_upload(uploaded_file, max_file_size_mb):
    """Process the uploaded file and split it into documents."""
    if len(uploaded_file.getvalue()) > max_file_size_mb * 1024 * 1024:
        st.error("File too large! Please upload a file smaller than 5MB.")
        return None

    try:
        file_content = uploaded_file.getvalue().decode("utf-8")
        return [Document(page_content=file_content)]
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None