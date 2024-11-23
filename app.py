import os
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

# Constants
MAX_FILE_SIZE_MB = 5
SAFE_RESPONSE_LIMIT = 2000
MAX_NEW_TOKENS = 250
SIMILARITY_THRESHOLD = 0.75  # Threshold for filtering low-similarity documents
TOKEN_LIMIT = 512  # Maximum tokens for context passed to the LLM

# Initialize session state for console
if "console" not in st.session_state:
    st.session_state.console = ""

# Function to update the console
def update_console(message):
    """Update the console with new messages."""
    st.session_state.console += f"{message}\n"

# Set up Streamlit app
st.set_page_config(page_title="Optimized RAG System", layout="wide")
st.title("Optimized RAG System with LangChain")
st.sidebar.header("Settings")

st.sidebar.info(
    """
    This app demonstrates a highly optimized Retrieval-Augmented Generation (RAG) system.
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
            update_console(f"File content length: {len(file_content)}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            update_console(f"Error loading file: {e}")

        # Splitting documents into chunks
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=150,  # Increased overlap for better context continuity
                separators=["\n\n", ".", "\n"]
            )
            texts = text_splitter.split_documents(documents)
            update_console(f"Text successfully split into {len(texts)} chunks.")
        except Exception as e:
            st.error(f"Error splitting text: {e}")
            update_console(f"Error splitting text: {e}")

        # Generating embeddings and storing them
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
            store = Chroma.from_documents(
                texts,
                embeddings,
                collection_name="uploaded_docs"
            )
            update_console(f"Embeddings generated and stored in Chroma. Total documents: {len(texts)}")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            update_console(f"Error generating embeddings: {e}")

        # Query Section
        query = st.text_input("Enter your query:", placeholder="Type your question here...")
        if query:
            try:
                update_console(f"Processing query: {query}")

                # Retrieve top-k most relevant documents
                retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                retrieved_docs = retriever.invoke(query)
                update_console(f"Retrieved {len(retrieved_docs)} documents.")

                # Filter low-similarity documents
                unique_docs = []
                for doc in retrieved_docs:
                    similarity_score = doc.metadata.get("similarity", 0)
                    if similarity_score >= SIMILARITY_THRESHOLD:
                        unique_docs.append(doc.page_content)

                update_console(f"Filtered to {len(unique_docs)} unique and relevant documents.")

                if not unique_docs:
                    st.warning("No relevant documents were found. Please try a different query.")
                    update_console("No relevant documents found for the query.")
                else:
                    # Aggregate context for LLM
                    context = "\n\n".join(unique_docs)
                    
                    # Ensure token limits
                    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
                    context_tokens = tokenizer(context)["input_ids"]
                    if len(context_tokens) > TOKEN_LIMIT:
                        context = tokenizer.decode(context_tokens[:TOKEN_LIMIT])
                        update_console("Context truncated due to token limits.")

                    formatted_query = (
                        f"Based on the following context:\n"
                        f"{context}\n\n"
                        f"Question: {query}\n\n"
                        f"Provide an answer strictly based on the context above. If the information is not available, respond with 'I do not know.'"
                    )
                    update_console(f"Formatted query: {formatted_query[:500]}...")  # Log truncated query for debugging

                    # Use InferenceClient for LLM
                    client = InferenceClient(model="google/flan-t5-large", token="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")
                    response = client.text_generation(
                        formatted_query,
                        max_new_tokens=MAX_NEW_TOKENS
                    )
                    update_console(f"Generated response: {response[:500]}")  # Log truncated response for debugging

                    # Limit response length
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
    height=300,
    key="console",
    disabled=True,
)
