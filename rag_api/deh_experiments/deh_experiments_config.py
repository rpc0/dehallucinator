SAMPLE_LDICTS_IDX = 0
SAMPLE_HYDE_LDICTS_IDX = 1
SAMPLE_SEMANTIC_LDICTS_IDX = 2

NO_RAG_NAIVE_PROMPT_IDX = 0
NO_RAG_NAIVE_DONT_LIE_PROMPT_IDX = 1
NO_RAG_PROMPT_IDX = 2
BASIC_RAG_S3_PROMPT_IDX = 3
BASIC_RAG_PROMPT_IDX = 4
BASIC_RAG_DONT_LIE_3SENTENCES_PROMPT_IDX = 5
BASIC_RAG_DONT_LIE_PROMPT_IDX = 6
BASIC_RAG_HYDE_NAIVE_PROMPT_IDX = 7
BASIC_RAG_HYDE_PROMPT_IDX = 8
BASIC_RAG_LLMAS_JUDGE_PROMPT_IDX = 9


# ======== Dictionary that contains all experiments definitions ==================
# Each experiment corresponds to an entry in this dictionary. An alternative to 
# this approach would be to read in experiments configuration information from 
# a text-based configuration file. Each experiment defintion is a dictionary itself.
# It contains the following entries:
#
# name:                       The name of the experiment (serves as an ID)
# title:                      The title (used for the histograms)
# sample_ldicts_idx           Index into the list of samples; selects the sample to be used for the experiment
# query_prompt_idx            Index into the list of prompts; selects the prompt to be used for the experiment
# context_needed              True, if context is needed (i.e. RAG), False else
# hyde_context_needed         True, if context based on Hyde is needed, False else
# suppress_answers            True, if LLM-as-judge is to be used; False else
# judges_suppress_threshold   Value between 0 and 1. answer is thrown away, if its final score falls below
# chunking_method             One of the following: "naive", "per_context", "per_article", "pseudo_semantic"
# vector_store_top_k          Number of documents to be retrieved from the vector store in similarity search
# llm_model                   Choice of LLM model among the available ones (use the name of the LLM)
# temperature                 Temperature setting for generating answers
#

experiments = {}


# ============================== Pass 1 ==========================================

experiments["P1_F1_K2"] = {
    "name": "P1_F1_K2",
    "title": "P1_F1_K2",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 2,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F1_K5"] = {
    "name": "P1_F1_K5",
    "title": "P1_F1_K5",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F1_K8"] = {
    "name": "P1_F1_K8",
    "title": "P1_F1_K8",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 8,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F1_K12"] = {
    "name": "P1_F1_K12",
    "title": "P1_F1_K12",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 12,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

# TODO: Rerun this experiment with context_needed set to True
experiments["P1_F4"] = {
    "name": "P1_F4",
    "title": "P1_F4",
    "sample_ldicts_idx": SAMPLE_HYDE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": False,
    "hyde_context_needed": True,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F6_per_context"] = {
    "name": "P1_F6_per_context",
    "title": "P1_F6_per_context",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "per_context",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F6_per_article"] = {
    "name": "P1_F6_per_article",
    "title": "P1_F6_per_article",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "per_article",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F6_pseudo_semantic"] = {
    "name": "P1_F6_pseudo_semantic",
    "title": "P1_F6_pseudo_semantic",
    "sample_ldicts_idx": SAMPLE_SEMANTIC_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "pseudo_semantic",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F7_50"] = {
    "name": "P1_F7_50",
    "title": "P1_F7_50",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    #"query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.5,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F7_60"] = {
    "name": "P1_F7_60",
    "title": "P1_F7_60",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    #"query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.6,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}


experiments["P1_F7_70"] = {
    "name": "P1_F7_70",
    "title": "P1_F7_70",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    #"query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F7_80"] = {
    "name": "P1_F7_80",
    "title": "P1_F7_80",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    #"query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.8,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F7_90"] = {
    "name": "P1_F7_90",
    "title": "P1_F7_90",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    #"query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.9,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["P1_F8_mistral"] = {
    "name": "P1_F8_mistral",
    "title": "P1_F8_mistral",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "mistral",
    "temperature": 0.5
}

experiments["P1_F8_gemma2_9b"] = {
    "name": "P1_F8_gemma2_9b",
    "title": "P1_F8_gemma2_9b",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "gemma2:9b",
    "temperature": 0.5
}

experiments["P1_F8_gemma2_27b"] = {
    "name": "P1_F8_gemma2_27b",
    "title": "P1_F8_gemma2_27b",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "gemma2:27b",
    "temperature": 0.5
}

experiments["P1_F8_qwen25_7b"] = {
    "name": "P1_F8_qwen25_7b",
    "title": "P1_F8_qwen25_7b",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P1_F8_qwen25_14b"] = {
    "name": "P1_F8_qwen25_14b",
    "title": "P1_F8_qwen25_14b",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:14b",
    "temperature": 0.5
}


experiments["P1_F9_00"] = {
    "name": "P1_F9_00",
    "title": "P1_F9_00",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.0
}

experiments["P1_F9_08"] = {
    "name": "P1_F9_08",
    "title": "P1_F9_08",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.8
}

experiments["P1_F9_20"] = {
    "name": "P1_F9_20",
    "title": "P1_F9_20",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 2.0
}

# ============================== PASS 2 =================================

experiments["P2_baseline"] = {
    "name": "P2_baseline",
    "title": "P2_baseline",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F1_K2"] = {
    "name": "P2_F1_K2",
    "title": "P2_F1_K2",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 2,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F1_K8"] = {
    "name": "P2_F1_K8",
    "title": "P2_F1_K8",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 8,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F1_K12"] = {
    "name": "P2_F1_K12",
    "title": "P2_F1_K12",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 12,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}


experiments["P2_F2"] = {
    "name": "P2_F2",
    "title": "P2_F2",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F3"] = {
    "name": "P2_F3",
    "title": "P2_F3",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_3SENTENCES_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F4"] = {
    "name": "P2_F4",
    "title": "P2_F4",
    "sample_ldicts_idx": SAMPLE_HYDE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": False,
    "hyde_context_needed": True,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}


experiments["P2_F6_per_context"] = {
    "name": "P2_F6_per_context",
    "title": "P2_F6_per_context",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "per_context",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F6_per_article"] = {
    "name": "P2_F6_per_article",
    "title": "P2_F6_per_article",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "per_article",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F6_pseudo_semantic"] = {
    "name": "P2_F6_pseudo_semantic",
    "title": "P2_F6_pseudo_semantic",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "pseudo_semantic",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F7_60"] = {
    "name": "P2_F7_60",
    "title": "P2_F7_60",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.6,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F7_70"] = {
    "name": "P2_F7_70",
    "title": "P2_F7_70",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F7_80"] = {
    "name": "P2_F7_80",
    "title": "P2_F7_80",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.8,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F7_90"] = {
    "name": "P2_F7_90",
    "title": "P2_F7_90",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.9,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["P2_F9_20"] = {
    "name": "P2_F9_20",
    "title": "P2_F9_20",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 2.0
}

experiments["P2_F9_08"] = {
    "name": "P2_F9_08",
    "title": "P2_F9_08",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.8
}

experiments["P2_F9_00"] = {
    "name": "P2_F9_00",
    "title": "P2_F9_00",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.0
}

# ========================= Final experiments =======================================================

experiments["FINAL_NO_RAG"] = {
    "name": "FINAL_NO_RAG",
    "title": "FINAL_NO_RAG",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": NO_RAG_NAIVE_PROMPT_IDX,
    "context_needed": False,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["FINAL_BASIC_RAG"] = {
    "name": "FINAL_BASIC_RAG",
    "title": "FINAL_BASIC_RAG",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_S3_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.7,    
    "chunking_method": "naive",
    "vector_store_top_k": 5,
    "llm_model": "llama3.1",
    "temperature": 0.5
}

experiments["FINAL_FINAL_RAG"] = {
    "name": "FINAL_FINAL_RAG",
    "title": "FINAL_FINAL_RAG",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": False,
    "judges_suppress_threshold": 0.8,    
    "chunking_method": "per_article",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["FINAL_FINAL_RAG_TR_60"] = {
    "name": "FINAL_FINAL_RAG_TR_60",
    "title": "FINAL_FINAL_RAG_TR_60",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.6,    
    "chunking_method": "per_article",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["FINAL_FINAL_RAG_TR_70"] = {
    "name": "FINAL_FINAL_RAG_TR_70",
    "title": "FINAL_FINAL_RAG_TR_70",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.7,
    "chunking_method": "per_article",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["FINAL_FINAL_RAG_TR_80"] = {
    "name": "FINAL_FINAL_RAG_TR_80",
    "title": "FINAL_FINAL_RAG_TR_80",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.8,
    "chunking_method": "per_article",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

experiments["FINAL_FINAL_RAG_TR_90"] = {
    "name": "FINAL_FINAL_RAG_TR_90",
    "title": "FINAL_FINAL_RAG_TR_90",
    "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
    "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
    "context_needed": True,
    "hyde_context_needed": False,
    "suppress_answers": True,
    "judges_suppress_threshold": 0.9,
    "chunking_method": "per_article",
    "vector_store_top_k": 5,
    "llm_model": "qwen2.5:7b",
    "temperature": 0.5
}

