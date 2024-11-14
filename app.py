import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Set the page title
st.set_page_config(page_title="Echo RAG", layout="wide")

# Title
st.title("Echo RAG")

# Sidebar
st.sidebar.header("Settings")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold", 0.0, 1.0, 0.5, step=0.05
)
st.sidebar.info(
    """
This is a simplified RAG system using sentence-transformers and sklearn for document retrieval.

**Powered by:** Sentence-Transformers
"""
)

@st.cache_resource
def load_embedder():
    logger.info("Loading Sentence Transformer model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model loaded successfully.")
    return embedder


embedder = load_embedder()

# Session state for uploaded documents and embeddings
if "documents" not in st.session_state:
    st.session_state["documents"] = None
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None

uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
if uploaded_file:
    # Delimiter selection
    delimiter_option = st.selectbox(
        "Select a delimiter to split the text into documents:",
        ["Linebreak (default)", "Comma", "Final Dots (Periods)", "Custom"],
    )
    if delimiter_option == "Linebreak (default)":
        delimiter = "\n\n"
    elif delimiter_option == "Comma":
        delimiter = ","
    elif delimiter_option == "Final Dots (Periods)":
        delimiter = "."
    else:  # Custom
        delimiter = st.text_input("Enter your custom delimiter:", value="")

    if delimiter and st.button("Process File"):
        try:
            # Read and normalize the uploaded file content
            text_data = uploaded_file.read().decode("utf-8")
            text_data = text_data.replace("\r\n", "\n").replace("\r", "\n")  # Normalize line breaks

            # Split documents
            documents = [doc.strip() for doc in text_data.split(delimiter) if doc.strip()]
            logger.info(f"Split into {len(documents)} documents using delimiter: '{delimiter}'")

            # Generate embeddings
            embeddings = embedder.encode(documents, convert_to_tensor=False)
            
            st.session_state["documents"] = documents
            st.session_state["embeddings"] = embeddings
            st.success(f"File successfully processed into {len(documents)} documents!")
            logger.info("Documents processed and embeddings generated.")
        except Exception as e:
            st.error(f"Failed to process file: {e}")
            logger.error(f"Error: {e}")

# Query processing
if st.session_state["documents"] is not None:
    st.header("Enter Your Query:")
    query = st.text_input("Query", placeholder="Type your question here...")
    if query:
        try:
            logger.info(f"Processing query: {query}")

            # Embed the query and ensure proper shape
            query_embedding = np.array(embedder.encode([query], convert_to_tensor=False)).reshape(1, -1)
            document_embeddings = np.array(st.session_state["embeddings"])

            # Compute similarities
            similarities = cosine_similarity(query_embedding, document_embeddings)[0]
            relevant_indices = [i for i, score in enumerate(similarities) if score >= similarity_threshold]
            relevant_docs = [(st.session_state["documents"][i], similarities[i]) for i in relevant_indices]

            # Display retrieved documents
            st.subheader("Retrieved Relevant Documents:")
            if relevant_docs:
                for idx, (doc, score) in enumerate(relevant_docs):
                    st.write(f"Document {idx+1} (Score: {score:.4f}):")
                    st.write(doc)
                    logger.info(f"Document {idx+1}: {doc} (Score: {score:.4f})")
            else:
                st.warning("No documents matched the similarity threshold.")
        except Exception as e:
            st.error(f"Failed to process query: {e}")
            logger.error(f"Query error: {e}")