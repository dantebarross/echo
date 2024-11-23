# modules/query_handler.py

from huggingface_hub import InferenceClient
from langchain.schema import Document
from transformers import AutoTokenizer
from modules.constants import MAX_NEW_TOKENS, TOKEN_LIMIT, RELEVANT_CONTEXT_RADIUS, HF_TOKEN, LLM_MODEL

# Initialize the Hugging Face inference client
client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

def retrieve_context(retriever, query, radius=RELEVANT_CONTEXT_RADIUS):
    """Retrieve and aggregate relevant context from documents."""
    print(f"Processing query: {query}")
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} documents.")
    
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i+1} content preview: {doc.page_content[:150]}...")
    
    # Aggregate context including related chunks
    aggregated_context = []
    for doc in retrieved_docs:
        content = doc.page_content
        match_index = content.lower().find(query.lower())
        if match_index != -1:
            # Include surrounding context
            start = max(0, match_index - radius)
            end = min(len(content), match_index + radius)
            snippet = content[start:end]
            aggregated_context.append(snippet)
            print(f"Snippet for match in doc {i+1}: {snippet[:150]}...")
        else:
            # Include entire content if no match
            aggregated_context.append(content[:radius * 2])
            print(f"Fallback snippet for doc {i+1}: {content[:150]}...")

    # Deduplicate and sort the context
    aggregated_context = list(dict.fromkeys(aggregated_context))  # Remove duplicates while preserving order
    context = "\n\n".join(aggregated_context)
    print(f"Final aggregated context length: {len(context)}")
    print(f"Aggregated context preview: {context[:500]}...")
    return context, retrieved_docs

def generate_answer(query, context):
    """Generate an answer using the Flan-T5 model with chain-of-thought prompting."""
    print(f"Generating answer for query: {query}")
    
    # Truncate context if needed
    context_tokens = tokenizer(context)["input_ids"]
    if len(context_tokens) > TOKEN_LIMIT:
        context = tokenizer.decode(context_tokens[:TOKEN_LIMIT])
        print("Context truncated due to token limits.")
    
    # Chain-of-thought prompt engineering
    formatted_query = (
        f"Context: {context}\n\n"
        f"Task: Based on the context provided, reason through the information step-by-step to answer the question.\n"
        f"Explain your reasoning if needed.\n\n"
        f"Question: {query}\n\n"
        f"Step-by-step response:"
    )

    print(f"Formatted query preview: {formatted_query[:500]}...")
    
    # Generate response
    response = client.text_generation(formatted_query, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Generated response: {response.strip()}")
    return response.strip()