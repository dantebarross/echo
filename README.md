# Echo RAG

## Overview
Echo RAG is a Retrieval-Augmented Generation (RAG) prototype designed to assist support agents in retrieving contextually relevant information from large datasets. This system uses a vector-based retrieval mechanism for document matching, combined with a streamlined interface to enable rapid knowledge access.

## Architecture
The system integrates key components to provide an efficient and user-friendly experience:

1. **Data Ingestion & Processing**: Allows users to upload `.txt` files, split the content into documents using customizable delimiters, and generate embeddings for each document using Sentence Transformers.
2. **Vector Database**: Leverages FAISS (Facebook AI Similarity Search) for efficient similarity-based document retrieval.
3. **User Interface**: A Streamlit-based UI facilitates query input, document retrieval, and review of results.

## Key Updates
- **File Support**: Users can now upload `.txt` files for processing, with options to split text by line breaks, commas, periods, or custom delimiters.
- **Relevant Document Retrieval**: The app retrieves all documents above a configurable similarity threshold, ensuring no limit on the number of relevant documents displayed.
- **Future Enhancements**: The final step, where a language model generates confident and contextually accurate answers, is identified as an area for improvement.

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

### 2. Prepare Data
Upload your `.txt` file through the application. A `mock.txt` file is available in the root folder as an example.

### 3. Launch the Application
Run the Streamlit application:
```bash
streamlit run app.py
```

## Features
- **Dynamic File Processing**: Supports `.txt` uploads and custom delimiter options for flexible data processing.
- **Semantic Search**: Utilizes Sentence Transformers and FAISS for efficient similarity-based document retrieval.
- **Configurable Similarity Threshold**: Adjustable threshold for fine-tuning document relevance.
- **User Feedback Mechanism**: Allows users to interact and provide feedback on retrieved documents.
- **Transparency**: Displays all retrieved documents for review, ensuring clarity and user trust.

## Example Usage
1. **Upload a File**: Upload a `.txt` file containing conversations or documents.
2. **Set Threshold**: Adjust the similarity threshold for filtering retrieved documents.
3. **Enter a Query**: Type a question or phrase to retrieve relevant documents.
4. **View Results**: The app displays all documents matching the threshold, allowing review of relevant context.

## Project Files
- `app.py`: Main application file handling data ingestion, retrieval, and display.
- `requirements.txt`: Lists dependencies for the project.

## Acknowledgments
Echo RAG demonstrates how RAG systems can streamline information retrieval and improve support workflows. While this version focuses on document retrieval, future updates will enhance response generation using contextually relevant information.

For questions or feedback, please contact **Danilo Barros**:
- LinkedIn: [https://www.linkedin.com/in/dantebarross/](https://www.linkedin.com/in/dantebarross/)