# modules/file_handler.py

from langchain.schema import Document
import streamlit as st
from modules.utils import update_console

def handle_file_upload(uploaded_file, max_file_size_mb):
    """Process the uploaded file and convert it to Document objects."""
    if len(uploaded_file.getvalue()) > max_file_size_mb * 1024 * 1024:
        st.error("File too large! Please upload a file smaller than 5MB.")
        update_console("File upload failed due to size limit.")
        return None

    try:
        file_content = uploaded_file.getvalue().decode("utf-8")
        update_console("File uploaded and processed successfully.")
        return [Document(page_content=file_content)]
    except Exception as e:
        st.error(f"Error loading file: {e}")
        update_console(f"Error processing uploaded file: {e}")
        return None