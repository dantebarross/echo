# modules/embeddings.py

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from modules.constants import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, HF_TOKEN

def split_documents(documents):
    """Split documents into smaller chunks with overlap."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Slightly larger chunks for more context
        chunk_overlap=200,  # More overlap to improve retrieval
        separators=["\n\n", ".", "\n"]
    )

    return text_splitter.split_documents(documents)

def create_vector_store(texts):
    """Generate embeddings and store them in a vector database."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(texts, embeddings, collection_name="uploaded_docs")