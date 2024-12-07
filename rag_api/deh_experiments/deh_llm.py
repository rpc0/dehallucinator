from langchain_ollama import ChatOllama

# ===================================================================
# LLM Parameters
LLM_MODEL_NAME_0 = "llama3.1"           
LLM_MODEL_NAME_1 = "mistral:latest"
LLM_MODEL_NAME_2 = "gemma2:27b"
LLM_MODEL_NAME_3 = "gemma2:9b"
LLM_MODEL_NAME_4 = "qwen2.5:14b"
LLM_MODEL_NAME_5 = "qwen2.5:7b"
#LLM_MODEL_NAME_6 = "llama3.3:latest"

CHAT_MODEL_NAME = LLM_MODEL_NAME_5

LLAMA3_1 = 0
MISTRAL_LATEST = 1
GEMMA2_27B = 2
GEMMA2_9B = 3
QWEN2_5_14B = 4
QWEN2_5_7B = 5

MAX_TOKENS = 20
TEMPERATURE = 0.0
TOP_P = 0.80
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0

llm_models = [LLM_MODEL_NAME_0, LLM_MODEL_NAME_1, LLM_MODEL_NAME_2,
              LLM_MODEL_NAME_3, LLM_MODEL_NAME_4, LLM_MODEL_NAME_5]

judge_llms = [MISTRAL_LATEST, GEMMA2_9B, QWEN2_5_7B]

JUDGES_SUPPRESS_THRESHOLD = 0.75


# ===================================================================
# Create the llm instance, based on the current query prompt
def get_llm(current_query_prompt, as_judge=False, judge_id=None):
    llm = ChatOllama(
        prompt_template=current_query_prompt,
        # model=CHAT_MODEL_NAME,
        model=llm_models[judge_id] if as_judge else CHAT_MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        gpu_use=True
    )

    return llm
