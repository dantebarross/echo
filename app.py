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
SIMILARITY_THRESHOLD = 0.75  # Initial threshold for filtering low-similarity documents
TOKEN_LIMIT = 512  # Maximum tokens for context passed to the LLM
RELEVANT_CONTEXT_RADIUS = 200  # Number of characters around the matched query

# Initialize session state for console
if "console" not in st.session_state:
    st.session_state.console = ""

# Function to update the console
def update_console(message):
    """Update the console with new messages."""
    st.session_state.console += f"{message}\n"

# Set up Streamlit app
st.set_page_config(page_title="Enhanced RAG System", layout="wide")
st.title("Enhanced Retrieval-Augmented Generation (RAG) System")
st.sidebar.header("Settings")

st.sidebar.info(
    """
    This app demonstrates a highly optimized Retrieval-Augmented Generation (RAG) system
    tailored for email-like documents and specific queries about detailed information.
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

        # Splitting documents into chunks with overlapping content
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,  # Smaller chunks for granular retrieval
                chunk_overlap=150,  # Overlap for better context
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
                retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.7})
                retrieved_docs = retriever.invoke(query)
                update_console(f"Retrieved {len(retrieved_docs)} documents.")

                # Gradual threshold relaxation for relevance
                thresholds = [0.75, 0.6, 0.5]  # Gradual thresholds
                relevant_docs = []
                for threshold in thresholds:
                    relevant_docs = [
                        doc for doc in retrieved_docs
                        if doc.metadata.get("similarity", 0) >= threshold
                    ]
                    if relevant_docs:
                        update_console(f"Documents found with threshold {threshold}.")
                        break

                # Fallback to top-K retrieval if no relevant documents are found
                if not relevant_docs:
                    update_console("No documents passed the threshold; using top-K fallback retrieval.")
                    relevant_docs = retrieved_docs[:3]

                # Aggregate context from relevant documents
                aggregated_context = []
                for doc in relevant_docs:
                    content = doc.page_content
                    match_index = content.lower().find(query.lower())  # Locate the query in the text
                    if match_index != -1:
                        # Include surrounding context around the match
                        start = max(0, match_index - RELEVANT_CONTEXT_RADIUS)
                        end = min(len(content), match_index + RELEVANT_CONTEXT_RADIUS)
                        snippet = content[start:end]
                        aggregated_context.append(snippet)
                    else:
                        # If no direct match, include the first part of the document
                        aggregated_context.append(content[:RELEVANT_CONTEXT_RADIUS * 2])

                # Remove duplicates
                aggregated_context = list(set(aggregated_context))
                context = "\n\n".join(aggregated_context)
                update_console(f"Aggregated context size: {len(context)} characters.")

                # Token truncation
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
                context_tokens = tokenizer(context)["input_ids"]
                if len(context_tokens) > TOKEN_LIMIT:
                    context = tokenizer.decode(context_tokens[:TOKEN_LIMIT])
                    update_console("Context truncated due to token limits.")

                # Format the query for LLM
                formatted_query = (
                    f"Based on the following context:\n"
                    f"{context}\n\n"
                    f"Question: {query}\n\n"
                    f"Provide an answer strictly based on the context above. If the information is not available, respond with 'I do not know.'"
                )
                update_console(f"Formatted query: {formatted_query[:500]}...")  # Debugging

                # Use InferenceClient for LLM
                client = InferenceClient(model="google/flan-t5-large", token="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")
                response = client.text_generation(
                    formatted_query,
                    max_new_tokens=MAX_NEW_TOKENS
                )
                update_console(f"Generated response: {response[:500]}")  # Log response

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
