from huggingface_hub import InferenceClient
from langchain.schema import Document
from transformers import AutoTokenizer
from modules.constants import MAX_NEW_TOKENS, TOKEN_LIMIT, RELEVANT_CONTEXT_RADIUS, HF_TOKEN, LLM_MODEL

# Initialize the Hugging Face inference client
client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

def retrieve_context(retriever, query, radius=RELEVANT_CONTEXT_RADIUS):
    """Retrieve and aggregate relevant context from documents."""
    try:
        retrieved_docs = retriever.invoke(query)
        aggregated_context = []
        for doc in retrieved_docs:
            content = doc.page_content
            match_index = content.lower().find(query.lower())
            if match_index != -1:
                start = max(0, match_index - radius)
                end = min(len(content), match_index + radius)
                snippet = content[start:end]
                aggregated_context.append(snippet)
            else:
                aggregated_context.append(content[:radius * 2])
        aggregated_context = list(dict.fromkeys(aggregated_context))
        context = "\n\n".join(aggregated_context)
        return context, retrieved_docs
    except Exception as e:
        raise RuntimeError(f"Error retrieving context: {e}")

def generate_answer(query, context):
    """Generate an answer using the Flan-T5 model."""
    try:
        context_tokens = tokenizer(context)["input_ids"]
        if len(context_tokens) > TOKEN_LIMIT:
            context = tokenizer.decode(context_tokens[:TOKEN_LIMIT])
        formatted_query = (
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Step-by-step response:"
        )
        response = client.text_generation(formatted_query, max_new_tokens=MAX_NEW_TOKENS)
        return response.strip()
    except Exception as e:
        raise RuntimeError(f"Error generating answer: {e}")