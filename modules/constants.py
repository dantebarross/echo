# modules/constants.py

import os

MAX_FILE_SIZE_MB = 5
SAFE_RESPONSE_LIMIT = 2000
MAX_NEW_TOKENS = 250
SIMILARITY_THRESHOLD = 0.75
TOKEN_LIMIT = 512
RELEVANT_CONTEXT_RADIUS = 200  # Number of characters around the matched query
CHUNK_SIZE = 400
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
LLM_MODEL = "google/flan-t5-large"

# Fetch the token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")