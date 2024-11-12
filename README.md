# Echo Framework

## Overview

Echo Framework is a prototype system for contextual data retrieval and response generation, aiming to streamline information access and improve workflow efficiency. This repository demonstrates a proof of concept using a vector-based retrieval system combined with an AI-based response generator, ideal for complex data environments.

## Architecture

This framework retrieves contextually relevant data from stored information and generates responses based on the query's context. It includes the following components:

1. **Data Ingestion & Processing**: Integrates data sources, preprocesses text, and generates embeddings for semantic retrieval.
2. **Vector Database**: Uses FAISS for efficient, similarity-based document retrieval.
3. **Language Model**: Generates contextually aware responses using open-source models like GPT-2 or GPT-Neo.
4. **User Interface**: Simple UI built with Streamlit for entering queries and viewing generated responses and sources.

## Getting Started

To get started, clone the repository and follow these steps:

### 1. Setup and Install Requirements

- **Create a Virtual Environment**:
  ```bash
  python -m venv echo_env
  source echo_env/bin/activate  # On Windows, use `echo_env\Scripts\activate`
  ```

- **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Prepare Mock Data

Create a `mock_data.csv` file with sample content:
```csv
id,text
1,"Sample text for testing retrieval and generation."
2,"Additional sample document content."
```

### 3. Generate Embeddings

Run `build_vector_db.py` to vectorize data for retrieval:
```bash
python build_vector_db.py
```

### 4. Launch the Application

Run the app locally with Streamlit:
```bash
streamlit run app.py
```

### 5. Deploying Online (Optional)

For online deployment, push the code to GitHub and use [Streamlit Community Cloud](https://streamlit.io/cloud) for easy, free hosting.

## Sample Workflow

1. **Enter a Query**: Type a query in the interface.
2. **Retrieve Relevant Documents**: The system retrieves top matches from the database.
3. **Generate Response**: The language model synthesizes a response based on retrieved content.
4. **Display Response and Sources**: Results and source documents are displayed for verification.
