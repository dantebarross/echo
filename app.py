import streamlit as st
import pandas as pd
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from streamlit_agraph import agraph, Node, Edge, Config
import warnings

# Set the page title
st.set_page_config(page_title="Echo")

# Suppress specific warnings from PyTorch and Transformers
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`")

# Title
st.title('Echo')

# Sidebar - Settings
st.sidebar.header('Settings')
k = st.sidebar.slider('Number of Documents to Retrieve', 1, 10, 3)
max_length = st.sidebar.slider('Maximum Response Length', 50, 500, 200)
use_sampling = st.sidebar.checkbox("Enable Sampling for Response Generation", value=False)

# Sidebar - About (moved below settings)
st.sidebar.title('About Echo')
st.sidebar.info('''
Echo is a prototype of a Retrieval-Augmented Generation (RAG) system designed to assist support agents.

**Features:**
- Semantic search with FAISS and Sentence Transformers.
- Response generation with Flan-T5.
- Adjustable settings.
- Feedback mechanism.

**Developed by:** Danilo Barros
https://www.linkedin.com/in/dantebarross/
''')

# Check if embeddings and FAISS index files exist
def check_files():
    return os.path.exists("embeddings.npy") and os.path.exists("vector.index")

# Function to create embeddings and build FAISS index if files are missing
def build_vector_db():
    # Check if the mock data file exists
    if not os.path.exists("mock_data.csv"):
        st.warning("Please upload a CSV file named 'mock_data.csv' to continue.")
        return None, None, None
    
    # Generate embeddings and FAISS index
    data = pd.read_csv("mock_data.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(data['text'].tolist())

    # Save embeddings and build FAISS index
    np.save("embeddings.npy", embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "vector.index")
    return data, embeddings, index

# File uploader to upload mock_data.csv
uploaded_file = st.file_uploader("Upload mock_data.csv", type="csv")
if uploaded_file:
    with open("mock_data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Uploaded successfully. Generating embeddings and index...")
    data, embeddings, index = build_vector_db()
else:
    # Check if files exist or trigger build if they don't
    if check_files():
        @st.cache_data
        def load_data():
            data = pd.read_csv('mock_data.csv')
            embeddings = np.load('embeddings.npy')
            index = faiss.read_index('vector.index')
            return data, embeddings, index
        data, embeddings, index = load_data()
    else:
        st.warning("Upload a CSV file named 'mock_data.csv' to initialize the application.")
        st.stop()

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy=False)
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

# User input
st.header('Enter Your Query:')
query = st.text_input('Query', placeholder='Type your question here...', label_visibility='collapsed')

if query:
    try:
        # Generate query embedding
        query_embedding = embedder.encode([query])

        # Retrieve relevant documents
        distances, indices = index.search(query_embedding, k)
        retrieved_docs = data.iloc[indices[0]]['text'].tolist()

        # Few-shot examples to guide the model's response
        few_shot_examples = """
        Example 1:
        Context:
        The app freezes when I try to play a video.
        Question:
        Why does the app freeze during video playback?
        Answer:
        The freezing issue might be due to an outdated video driver or limited memory. Please try updating your drivers and close unnecessary applications to free up memory.

        Example 2:
        Context:
        I'm unable to log in to the app.
        Question:
        What should I do if I can't log in?
        Answer:
        Please check your internet connection and ensure you're using the correct password. If you've forgotten your password, click on 'Forgot Password' to reset it.

        Example 3:
        Context:
        The app is running slowly.
        Question:
        How can I improve app performance?
        Answer:
        Try clearing the app cache and restarting your device. If the issue persists, consider updating to the latest app version or reinstalling the app.
        """

        # Prepare the prompt for the language model with few-shot examples
        context = "\n".join(retrieved_docs)
        prompt = f"""{few_shot_examples}

        Context:
        {context}

        Question:
        {query}

        Answer:"""

        # Generate response
        inputs = tokenizer(prompt, return_tensors='pt')
        
        if use_sampling:
            # Sampling-based generation for more varied responses
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                early_stopping=True
            )
        else:
            # Deterministic generation
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.replace("Answer:", "").strip()  # Remove "Answer:" if it appears in output

        # Display response and sources
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Generated Response:')
            st.write(response)

            st.subheader('Was this answer helpful?')
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                if st.button('👍 Yes'):
                    st.success('Thank you for your feedback!')
            with feedback_col2:
                if st.button('👎 No'):
                    st.error('We appreciate your feedback and will work to improve.')

        with col2:
            st.subheader('Retrieved Documents:')
            for idx, doc in enumerate(retrieved_docs):
                with st.expander(f"Document {idx+1}"):
                    st.write(doc)

        # Application Flow Diagram
        st.subheader('Application Flow Diagram')

        # Define sub-nodes for detailed flow representation
        nodes = [
            Node(id="User_Input", label="User Input", shape="box", color="lightblue"),
            Node(id="User_Display", label="User Display", shape="box", color="lightblue"),
            Node(id="App", label="Streamlit App", shape="box", color="lightblue"),
            Node(id="VectorDB_Query", label="Vector Database Query", shape="box", color="lightblue"),
            Node(id="VectorDB_Result", label="Vector Database Result", shape="box", color="lightblue"),
            Node(id="Model_Generate", label="Language Model Generate", shape="box", color="lightblue"),
            Node(id="Model_Response", label="Language Model Response", shape="box", color="lightblue"),
        ]

        # Define detailed edges with clear labels
        edges = [
            Edge(source="User_Input", target="App", label="Send Query"),
            Edge(source="App", target="VectorDB_Query", label="Retrieve Docs"),
            Edge(source="VectorDB_Query", target="VectorDB_Result", label="Process Query"),
            Edge(source="VectorDB_Result", target="App", label="Return Docs"),
            Edge(source="App", target="Model_Generate", label="Prepare Response"),
            Edge(source="Model_Generate", target="Model_Response", label="Generate Text"),
            Edge(source="Model_Response", target="App", label="Return Response"),
            Edge(source="App", target="User_Display", label="Display Answer"),
        ]

        # Adjust the config for better readability
        config = Config(
            width=1200,
            height=800,
            directed=True,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=True,
            nodeDistanceMultiplier=2.5,  # Adjust distance between nodes
            linkDistance=350,  # Increase space for links
            edgeLabelPosition='middle',  # Center edge labels
            edgeFontSize=12,  # Font size for edges
            edgeFontColor='#333333'  # Dark color for contrast
        )
        
        agraph(nodes=nodes, edges=edges, config=config)

    except Exception as e:
        st.error(f"An error occurred: {e}")