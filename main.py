from transformers import pipeline

########################################
# 1. Pipeline Setup with Prompt Prefix
########################################

# short “system-style” instruction for chit-chat
CHITCHAT_PROMPT_PREFIX = (
    "You are a friendly, helpful assistant. Respond as yourself. DO NOT repeat the user's question, input, or yourself entirely."
    "Provide a concise, relevant, and warm reply.\n\n"
)

# chit-chat pipeline
chat_pipeline = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    max_length=100,
    truncation=True,
    no_repeat_ngram_size=3  # Avoid repeating 3-gram sequences
)

# a simple QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

########################################
# 2. Context for QA
########################################
FINANCE_CONTEXT = (
    "Current mortgage interest rates vary depending on the economy, "
    "loan type, credit score, and lender. They can range from around 3% "
    "to 7%. Investing money often involves strategies like diversifying "
    "in stocks, bonds, or mutual funds, and depends on one's risk tolerance."
)

########################################
# 3. Intent Classification (Rule-Based)
########################################

def classify_intent(user_input: str) -> str:
    domain_keywords = [
        "finance", "loan", "interest", "apy", "apr",
        "stocks", "invest", "money", "bank", "mortgage"
    ]
    text = user_input.lower()
    if any(kw in text for kw in domain_keywords):
        return "informative"
    return "chitchat"

########################################
# 4. Response Functions
########################################

def respond_informative(user_input: str) -> str:
    """
    Use the QA pipeline with a known context for finance-related queries.
    """
    result = qa_pipeline(question=user_input, context=FINANCE_CONTEXT)
    return f"[HF-QA] {result['answer']}"

def respond_chitchat(user_input: str) -> str:
    """
    Use the chat pipeline with a prompt prefix for more natural conversation.
    """
    # Combine the prefix and user’s question
    prompt = (
        CHITCHAT_PROMPT_PREFIX
        + f"User: {user_input}\n"
        + "ChitChat:"
    )

    
    result = chat_pipeline(
        prompt,
        min_new_tokens = 10 #adjust accordingly for response length
        #idea for bigger prototype: have different token lengths for casual and informative?
    )
    

    generated_text = result[0]["generated_text"]
    
    if "ChitChat:" in generated_text:
        chitChat_response = generated_text.split("ChitChat:")[-1].strip()
    else:
        chitChat_response = generated_text
    
    return f"[HF-Chat] {chitChat_response}"

########################################
# 5. Router
########################################
def handle_user_query(user_input: str) -> str:
    intent = classify_intent(user_input)
    if intent == "informative":
        return respond_informative(user_input)
    else:
        return respond_chitchat(user_input)

########################################
# 6. Demo
########################################
if __name__ == "__main__":
    user_queries = [
        "Hey there, how's life?",
        "What's the best mortgage interest rate?",
        "Any tips on investing my money?",
        "Hello, how are you doing?"
    ]
    
    for q in user_queries:
        print(f"User: {q}")
        response = handle_user_query(q)
        print(f"TempChitChat: {response}\n")
        