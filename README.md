# Echo RAG

## Overview
Echo RAG is a Retrieval-Augmented Generation (RAG) prototype designed to assist support agents in retrieving contextually relevant information from large datasets. This system uses a vector-based retrieval mechanism for document matching, combined with a streamlined interface to enable rapid knowledge access and response generation.

## Architecture
The system integrates key components to provide an efficient and user-friendly experience:

1. **Data Ingestion & Processing**:
   - Supports `.txt` file uploads.
   - Splits content into documents using customizable delimiters and chunking with overlap.
   - Generates embeddings for documents using Hugging Face Sentence Transformers.

2. **Vector Database**:
   - Leverages ChromaDB for efficient similarity-based document retrieval.

3. **Query Handling**:
   - Retrieves context using similarity-based methods.
   - Aggregates relevant chunks with fine-tuned controls for query-context alignment.

4. **User Interface**:
   - A Streamlit-based UI facilitates file uploads, query input, and viewing of results.
   - Console logs for debugging and transparency.

5. **Answer Generation**:
   - Employs Hugging Face models for generating accurate, step-by-step answers based on retrieved context.

## Key Updates
- **Environment Variables**: Hugging Face token (`HF_TOKEN`) is securely managed using environment variables, making the system more secure and flexible.
- **Dynamic Chunk Splitting**: Documents are split into chunks with customizable size and overlap, ensuring better retrieval performance.
- **Context Aggregation**: Enhanced query handling includes retrieving and deduplicating relevant snippets around matched queries.
- **Embeddings Storage**: Uses ChromaDB for robust vector storage and retrieval.
- **Improved Modularity**: All major functionalities are organized into modules (`file_handler`, `query_handler`, `vector_store`, `utils`, etc.) for better scalability and maintainability.

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

### 2. Set Environment Variables

Create a `.streamlit/secrets.toml` file to securely store the Hugging Face token:
```toml
HF_TOKEN = "your_huggingface_token_here"
```

Alternatively, export the token as an environment variable:
```bash
export HF_TOKEN="your_huggingface_token_here"  # For Linux/macOS
set HF_TOKEN="your_huggingface_token_here"    # For Windows
```

### 3. Launch the Application
Run the Streamlit application:
```bash
streamlit run app.py
```

### 4. Prepare Data
Upload your `.txt` file through the application. Example text files can be used for testing.

## Features
- **Secure Token Management**: Hugging Face token is fetched from environment variables for enhanced security.
- **Dynamic File Processing**:
  - Supports `.txt` uploads.
  - Customizable chunk size and overlap for efficient document splitting.
- **Semantic Search**:
  - Uses Sentence Transformers and ChromaDB for robust similarity-based document retrieval.
- **Configurable Context Retrieval**:
  - Aggregates context snippets dynamically around query matches.
  - Adjustable similarity parameters.
- **Answer Generation**:
  - Generates step-by-step answers using Hugging Face LLMs with chain-of-thought prompting.
- **Debugging Console**:
  - Provides detailed logs for transparency and troubleshooting.

## Example Usage
1. **Upload a File**: Upload a `.txt` file containing documents or text data.
2. **View Console Logs**: Check processing and retrieval updates in the console area.
3. **Enter a Query**: Type a question or phrase to retrieve relevant context.
4. **View Results**: The app displays all relevant documents and generates a detailed answer.

## Project Files
- `app.py`: Main application file for UI and core workflows.
- `modules/`: Contains modularized components:
  - `file_handler.py`: Handles file uploads and document processing.
  - `vector_store.py`: Manages document chunking and vector database storage.
  - `query_handler.py`: Processes queries and generates responses.
  - `utils.py`: Utility functions, including console logging.
  - `constants.py`: Project-wide constants.
- `.streamlit/secrets.toml`: Stores environment variables for secure configuration.

## Clearing Cache
To clear cached Hugging Face models and datasets:
```bash
huggingface-cli delete-cache
```

Alternatively, manually delete the cache directory:
- **Linux/macOS**: `~/.cache/huggingface`
- **Windows**: `C:\Users\<YourUsername>\.cache\huggingface`

## Acknowledgments
Echo RAG demonstrates how RAG systems can streamline information retrieval and improve workflows. This version integrates robust document retrieval and context aggregation, with plans to enhance the response generation module further.

For questions or feedback, please contact **Danilo Barros**:
- LinkedIn: [https://www.linkedin.com/in/dantebarross/](https://www.linkedin.com/in/dantebarross/)