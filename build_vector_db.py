# This script is crucial for enabling efficient data retrieval in the RAG (Retrieval-Augmented Generation) 
# system by transforming support-related text data into vector embeddings and indexing them. 
# It reads mock_data.csv, uses the all-MiniLM-L6-v2 model from Sentence Transformers 
# to encode each text entry into a high-dimensional vector, and saves these embeddings 
# along with unique IDs for easy access. The embeddings are then organized into a FAISS index (vector.index), 
# which allows the system to quickly retrieve contextually relevant information based on semantic similarity. 
# This setup is essential for providing fast, accurate responses in support scenarios 
# by retrieving the most relevant data for each query.

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load mock data
data = pd.read_csv('mock_data.csv')

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(data['text'].tolist())

# Save embeddings and IDs
np.save('embeddings.npy', embeddings)
data['id'].to_csv('ids.csv', index=False)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index
faiss.write_index(index, 'vector.index')