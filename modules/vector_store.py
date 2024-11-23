# modules/vector_store.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from modules.constants import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from modules.utils import update_console
import faiss

def split_documents(documents):
    """Split documents into smaller chunks with overlap."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", ".", "\n"]
        )
        chunks = text_splitter.split_documents(documents)
        update_console(f"Split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        update_console(f"Error splitting documents: {e}")
        raise

def create_vector_store(texts):
    """Generate embeddings and store them in a FAISS vector database."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Set up retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        update_console("Vector store created successfully with FAISS.")
        return vector_store, retriever
    except Exception as e:
        update_console(f"Error creating vector store: {e}")
        raise