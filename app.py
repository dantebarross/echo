import os
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import numpy as np

# Constants
MAX_FILE_SIZE_MB = 5
CLUSTER_THRESHOLD = 0.7  # Similarity threshold for clustering
SAFE_RESPONSE_LIMIT = 2000  # Safe character limit for responses
MAX_NEW_TOKENS = 250  # Hugging Face limit

# Initialize session state for console
if "console" not in st.session_state:
    st.session_state.console = ""

# Function to update the console
def update_console(message):
    """Update the console with new messages."""
    st.session_state.console += f"{message}\n"

# Set up Streamlit app
st.set_page_config(page_title="LangChain RAG System", layout="wide")
st.title("LangChain-powered RAG System with Clustering")
st.sidebar.header("Settings")

st.sidebar.info(
    """
    This app demonstrates a Retrieval-Augmented Generation (RAG) pipeline with clustering for better responses.

    **Powered by:** LangChain, Hugging Face, and ChromaDB.
    """
)

# File upload section
uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
if uploaded_file:
    if len(uploaded_file.getvalue()) > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error("File too large! Please upload a file smaller than 5MB.")
    else:
        update_console("File loaded successfully!")
        try:
            file_content = uploaded_file.getvalue().decode("utf-8")
            documents = [Document(page_content=file_content)]
        except Exception as e:
            st.error(f"Error loading file: {e}")
            update_console(f"Error loading file: {e}")

        update_console("Splitting documents into chunks...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["From:", "Subject:", "\n\n", "\n", "."]
            )
            texts = text_splitter.split_documents(documents)
            update_console(f"Text successfully split into {len(texts)} chunks!")
        except Exception as e:
            st.error(f"Error splitting text: {e}")
            update_console(f"Error splitting text: {e}")

        update_console("Generating embeddings and storing in Chroma...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            store = Chroma.from_documents(
                texts,
                embeddings,
                collection_name="uploaded_docs"
            )
            update_console("Embeddings generated and stored in Chroma!")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            update_console(f"Error generating embeddings: {e}")

        # Query Section
        query = st.text_input("Enter your query:", placeholder="Type your question here...")
        if query:
            try:
                update_console(f"Processing query: {query}")
                retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                retrieved_docs = retriever.get_relevant_documents(query)

                # Aggregate context from retrieved documents
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                formatted_query = (
                    f"Based on the following context:\n"
                    f"{context}\n\n"
                    f"Question: {query}\n\n"
                    f"Provide a detailed and actionable answer."
                )

                # Use InferenceClient for LLM
                client = InferenceClient(model="google/flan-t5-large", token="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")
                response = client.text_generation(
                    formatted_query,
                    max_new_tokens=MAX_NEW_TOKENS
                )

                # Ensure a complete response
                if len(response) > SAFE_RESPONSE_LIMIT:
                    response = response[:SAFE_RESPONSE_LIMIT] + "... (truncated)"

                update_console("Query processed successfully!")
                st.subheader("Answer:")
                st.write(response)

            except Exception as e:
                st.error(f"Error processing query: {e}")
                update_console(f"Error processing query: {e}")

# Render the console area
st.text_area(
    "Console",
    value=st.session_state.console,
    height=100,  # Smaller height (4 lines approx)
    key="console",
    disabled=True,  # Prevent user edits
)
