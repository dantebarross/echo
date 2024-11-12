# Echo

## Overview

Echo is a proof-of-concept Retrieval-Augmented Generation (RAG) system designed to assist support agents in retrieving contextually relevant information and generating accurate responses based on a user’s query. This system uses a vector-based retrieval mechanism for document matching combined with a language model for generating responses, providing a streamlined experience for accessing support information.

## Architecture

The RAG system combines several core components to provide efficient and context-aware response generation:

1. **Data Ingestion & Processing**: Loads and processes mock data, converts text to embeddings using Sentence Transformers, and stores the embeddings in a vector database.
2. **Vector Database**: Uses FAISS (Facebook AI Similarity Search) for fast, similarity-based retrieval of relevant documents.
3. **Language Model**: Employs the FLAN-T5 language model for generating responses that are contextually aligned with the retrieved documents.
4. **User Interface**: A Streamlit-based interface allows users to input queries, view generated responses, and examine source documents for transparency.

## Getting Started

To set up and run the application locally, follow these steps:

### 1. Setup and Install Requirements

- **Create a Virtual Environment**:
  ```bash
  python -m venv rag_env
  source rag_env/bin/activate  # On Windows, use `rag_env\Scripts\activate`
  ```

- **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Prepare Mock Data

Create a `mock_data.csv` file in the root directory with sample content:
```csv
id,text
1,"Sample text for testing retrieval and generation."
2,"Additional sample document content for retrieval."
```

### 3. Generate Embeddings

To build the vector database from mock data, run `build_vector_db.py`:
```bash
python build_vector_db.py
```
This script will:
- Encode text entries in `mock_data.csv` into embeddings.
- Store the embeddings in a FAISS index (`vector.index`) for efficient retrieval.

### 4. Launch the Application

Run the Streamlit application locally:
```bash
streamlit run app.py
```

### 5. Optional: Online Deployment

To deploy the application online, push the code to GitHub and use [Streamlit Community Cloud](https://streamlit.io/cloud) for free and easy hosting.

## Sample Workflow

1. **Enter a Query**: Type a question or query in the text input field on the Streamlit app.
2. **Retrieve Relevant Documents**: The system searches the vector database and retrieves the top-matching documents based on similarity.
3. **Generate Response**: The FLAN-T5 language model generates a response using the retrieved documents as context.
4. **Display Response and Sources**: The generated response is displayed along with the retrieved documents, allowing users to verify the response's accuracy.

## Features

- **Semantic Search**: Uses Sentence Transformers for creating embeddings and FAISS for similarity search.
- **Context-Aware Response Generation**: Leverages the FLAN-T5 language model for generating responses based on the query and document context.
- **Adjustable Settings**: Allows configuration of the number of documents to retrieve and the maximum response length.
- **User Feedback**: Provides a feedback mechanism to gauge response accuracy.
- **Application Flow Diagram**: Displays a flowchart to help users understand the system's internal flow.

## Example Usage

1. **Setup**: Ensure the application dependencies are installed and the mock data is processed.
2. **Run the App**: Start the application with `streamlit run app.py`.
3. **Enter a Query**: Enter questions like "Why does the app freeze during video playback?" or "How can I improve app performance?" to see responses generated based on the provided data.
4. **View Responses**: The app displays the generated response and the retrieved documents, allowing you to validate the information source.

## Project Files

- `app.py`: The main Streamlit application file, handling user input, document retrieval, and response generation.
- `build_vector_db.py`: Script for building the FAISS vector database from mock data.
- `mock_data.csv`: Sample data file containing text entries for testing the retrieval and response generation system.

## Requirements

- `streamlit`
- `faiss`
- `pandas`
- `numpy`
- `transformers`
- `sentence-transformers`
- `streamlit-agraph`

## Acknowledgments

This project demonstrates a simple yet powerful RAG-based system, using open-source tools and models to showcase how context-aware retrieval and response generation can enhance support workflows.

For any questions or feedback, please contact **Danilo Barros**:
- LinkedIn: [https://www.linkedin.com/in/dantebarross/](https://www.linkedin.com/in/dantebarross/)
