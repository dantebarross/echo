import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file (if available)
load_dotenv()

MAX_FILE_SIZE_MB = 5
SAFE_RESPONSE_LIMIT = 2000
MAX_NEW_TOKENS = 250
SIMILARITY_THRESHOLD = 0.75
TOKEN_LIMIT = 512
RELEVANT_CONTEXT_RADIUS = 200
CHUNK_SIZE = 400
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
LLM_MODEL = "google/flan-t5-large"

# Attempt to fetch HF_TOKEN from .env file or system environment
HF_TOKEN = os.getenv("HF_TOKEN")

# If not set, fallback to Streamlit secrets (for deployed apps)
if not HF_TOKEN and "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN is not set. Ensure it's available in a .env file, as an environment variable, or in Streamlit Secrets."
    )