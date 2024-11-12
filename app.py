# This file builds the streamlit web application

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load data and index
data = pd.read_csv('mock_data.csv')
embeddings = np.load('embeddings.npy')
index = faiss.read_index('vector.index')

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load language model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# Streamlit app
st.title('RAG Prototype Demo')

# User input
query = st.text_input('Enter your query:')

if query:
    # Generate query embedding
    query_embedding = embedder.encode([query])

    # Retrieve relevant documents
    k = 3  # Number of documents to retrieve
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = data.iloc[indices[0]]['text'].tolist()

    # Prepare the prompt for the language model
    context = "\n".join(retrieved_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    # Generate response
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=150, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display response and sources
    st.subheader('Generated Response:')
    st.write(response.split('Answer:')[-1].strip())

    st.subheader('Retrieved Documents:')
    for doc in retrieved_docs:
        st.write(f"- {doc}")