from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from modules.constants import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from modules.utils import update_console

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
    """Generate embeddings and store them in a vector database."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = Chroma.from_documents(
            texts, embeddings, collection_name="uploaded_docs",
            persist_directory="./chroma_db"  # Ensure local persistence
        )
        vector_store.persist()  # Save the database
        update_console("Vector store created successfully.")
        return vector_store
    except Exception as e:
        update_console(f"Error creating vector store: {e}")
        raise