# app.py

import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

import streamlit as st
from modules.file_handler import handle_file_upload
from modules.vector_store import split_documents, create_vector_store
from modules.query_handler import retrieve_context, generate_answer
from modules.constants import MAX_FILE_SIZE_MB
from modules.utils import update_console

# Initialize session state
if "console" not in st.session_state:
    st.session_state.console = ""

st.set_page_config(page_title="Enhanced RAG System", layout="wide")
st.title("Enhanced Retrieval-Augmented Generation (RAG) System")
st.sidebar.header("Settings")

# File upload
uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
if uploaded_file:
    documents = handle_file_upload(uploaded_file, MAX_FILE_SIZE_MB)
    if documents:
        update_console("File loaded successfully!")
        try:
            # Process uploaded file
            texts = split_documents(documents)
            vector_store = create_vector_store(texts)
            retriever = vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.7}
            )
            update_console(f"Text successfully split into {len(texts)} chunks.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            update_console(f"Error: {e}")

        # Query input
        query = st.text_input("Enter your query:", placeholder="Type your question here...")
        if query:
            try:
                context, retrieved_docs = retrieve_context(retriever, query)
                response = generate_answer(query, context)
                st.subheader("Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {e}")
                update_console(f"Error: {e}")

st.text_area("Console", height=300, key="console", disabled=True)