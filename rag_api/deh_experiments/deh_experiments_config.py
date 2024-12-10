SAMPLE_LDICTS_IDX = 0
SAMPLE_HYDE_LDICTS_IDX = 1
SAMPLE_SEMANTIC_LDICTS_IDX = 2

NO_RAG_NAIVE_PROMPT_IDX = 0
NO_RAG_PROMPT_IDX = 1
BASIC_RAG_NAIVE_PROMPT_IDX = 2
BASIC_RAG_PROMPT_IDX = 3
BASIC_RAG_DONT_LIE_PROMPT_IDX = 4
BASIC_RAG_HYDE_PROMPT_IDX = 5
BASIC_RAG_SUPPRESS_ANSWSERS_PROMPT_IDX = 6

EXPERIMENT_GLOBAL_ID = "P0"

experiments = {}

RUN_ALL_EXPERIMENTS = False
experiments_include = {
    "NO_RAG_NAIVE": False,
    "NO_RAG": False,
    "BASIC_RAG_NAIVE": False,
    "BASIC_RAG": True,
    "BASIC_RAG_DONT_LIE": False,
    "BASIC_RAG_LLM_AS_JUDGE": False,
    "BASIC_RAG_HYDE": False,
    "BASIC_RAG_SEMANTIC_CHUNKING": False,
    "FULL_RAG": False
}

experiments["NO_RAG_NAIVE"] = {"name": "NO_RAG_NAIVE",
                         "title": f"{EXPERIMENT_GLOBAL_ID}_NO_RAG_NAIVE",
                         "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                         "query_prompt_idx": NO_RAG_NAIVE_PROMPT_IDX,
                         "context_needed": False,
                         "hyde_context_needed": False,
                         "suppress_answers": False}

experiments["NO_RAG"] = {"name": "NO_RAG",
                         "title": f"{EXPERIMENT_GLOBAL_ID}_NO_RAG",
                         "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                         "query_prompt_idx": NO_RAG_PROMPT_IDX,
                         "context_needed": False,
                         "hyde_context_needed": False,
                         "suppress_answers": False}

experiments["BASIC_RAG_NAIVE"] = {"name": "BASIC_RAG_NAIVE",
                            "title": f"{EXPERIMENT_GLOBAL_ID}_BASIC_RAG_NAIVE",
                            "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                            "query_prompt_idx": BASIC_RAG_NAIVE_PROMPT_IDX,
                            "context_needed": True,
                            "hyde_context_needed": False,
                            "suppress_answers": False}

experiments["BASIC_RAG"] = {"name": "BASIC_RAG",
                            "title": f"{EXPERIMENT_GLOBAL_ID}_BASIC_RAG",
                            "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                            "query_prompt_idx": BASIC_RAG_PROMPT_IDX,
                            "context_needed": True,
                            "hyde_context_needed": False,
                            "suppress_answers": False}

experiments["BASIC_RAG_DONT_LIE"] = {"name": "BASIC_RAG_DONT_LIE",
                                     "title": f"{EXPERIMENT_GLOBAL_ID}_BASIC_RAG_DONT_LIE",
                                     "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                                     "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
                                     "context_needed": True,
                                     "hyde_context_needed": False,
                                     "suppress_answers": False}

experiments["BASIC_RAG_LLM_AS_JUDGE"] = {"name": "BASIC_RAG_LLM_AS_JUDGE",
                                         "title": f"{EXPERIMENT_GLOBAL_ID}_BASIC_RAG_LLM_AS_JUDGE",
                                         "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                                         "query_prompt_idx": BASIC_RAG_PROMPT_IDX,
                                         "context_needed": True,
                                         "hyde_context_needed": False,
                                         "suppress_answers": True}

experiments["BASIC_RAG_HYDE"] = {"name": "BASIC_RAG_HYDE",
                                 "title": f"{EXPERIMENT_GLOBAL_ID}_BASIC_RAG_HYDE",
                                 "sample_ldicts_idx": SAMPLE_HYDE_LDICTS_IDX,
                                 "query_prompt_idx": BASIC_RAG_PROMPT_IDX,
                                 "context_needed": True,
                                 "hyde_context_needed": True,
                                 "suppress_answers": False}

experiments["BASIC_RAG_SEMANTIC_CHUNKING"] = {"name": "BASIC_RAG_SEMANTIC_CHUNKING",
                                              "title": f"{EXPERIMENT_GLOBAL_ID}_BASIC_RAG_SEMANTIC_CHUNKING",
                                              "sample_ldicts_idx": SAMPLE_SEMANTIC_LDICTS_IDX,
                                              "query_prompt_idx": BASIC_RAG_PROMPT_IDX,
                                              "context_needed": True,
                                              "hyde_context_needed": False,
                                              "suppress_answers": False}

experiments["FULL_RAG"] = {"name": "FULL_RAG",
                           "title": f"{EXPERIMENT_GLOBAL_ID}_FULL_RAG",
                           "sample_ldicts_idx": SAMPLE_SEMANTIC_LDICTS_IDX, # SAMPLE_HYDE_LDICTS_IDX,
                           "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
                           "context_needed": True,
                           "hyde_context_needed": False,
                           "suppress_answers": False}
