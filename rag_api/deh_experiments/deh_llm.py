from langchain_ollama import ChatOllama

# ===================================================================
# LLM Parameters
CHAT_MODEL_NAME = "llama3.1"
MAX_TOKENS = 100
TEMPERATURE = 0.5
TOP_P = 0.95
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0


# ===================================================================
# Create the llm instance, based on the current query prompt
def get_llm(current_query_prompt):
    llm = ChatOllama(
        prompt_template=current_query_prompt,
        model=CHAT_MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        gpu_use=True
    )

    return llm
