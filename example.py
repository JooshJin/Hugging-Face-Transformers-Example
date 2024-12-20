from transformers import pipeline

# Initialize Hugging Face pipelines
qa_pipeline = pipeline("question-answering")
chat_pipeline = pipeline("text-generation", model="gpt2")
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def classify_query(query: str) -> str:
    """
    Classify query as 'informational' or 'chit-chat'.
    """
    # Example: Classify based on tone or content
    result = classifier(query)
    if "informational" in result[0]['label'].lower():
        return "informational"
    return "chit-chat"

def respond_to_query(query: str):
    """
    Route query to the appropriate response pipeline.
    """
    query_type = classify_query(query)
    
    if query_type == "informational":
        # Example context for QA pipeline
        context = "Hugging Face provides tools for NLP tasks like sentiment analysis and text generation."
        response = qa_pipeline(question=query, context=context)
        return response['answer']
    else:
        # Generate chit-chat response
        response = chat_pipeline(query, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']

# Example user queries
queries = [
    "What is Hugging Face?",
    "How are you doing today?"
]

for q in queries:
    print(f"User: {q}")
    print(f"Assistant: {respond_to_query(q)}\n")

