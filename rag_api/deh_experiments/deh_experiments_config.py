SAMPLE_LDICTS_IDX = 0
SAMPLE_HYDE_LDICTS_IDX = 1
SAMPLE_SEMANTIC_LDICTS_IDX = 2

NAIVE_PROMPT_IDX = 0
BASIC_RAG_PROMPT_IDX = 1
BASIC_RAG_DONT_LIE_PROMPT_IDX = 2
BASIC_RAG_HYDE_PROMPT_IDX = 3
BASIC_RAG_SUPPRESS_ANSWSERS_PROMPT_IDX = 4

experiments = {}

RUN_ALL_EXPERIMENTS = False
experiments_include = {
    "NO_RAG": True,
    "BASIC_RAG": True,
    "BASIC_RAG_DONT_LIE": True,
    "BASIC_RAG_SUPPRESS_ANSWERS": False,
    "BASIC_RAG_HYDE": False,
    "BASIC_RAG_SEMANTIC_CHUNKING": True,
    "FULL_RAG": True
}

experiments["NO_RAG"] = {"name": "NO_RAG",
                         "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                         "query_prompt_idx": NAIVE_PROMPT_IDX,
                         "context_needed": False,
                         "hyde_context_needed": False,
                         "suppress_answers": False}

experiments["BASIC_RAG"] = {"name": "BASIC_RAG",
                            "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                            "query_prompt_idx": BASIC_RAG_PROMPT_IDX,
                            "context_needed": True,
                            "hyde_context_needed": False,
                            "suppress_answers": False}

experiments["BASIC_RAG_DONT_LIE"] = {"name": "BASIC_RAG_DONT_LIE",
                                     "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                                     "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
                                     "context_needed": True,
                                     "hyde_context_needed": False,
                                     "suppress_answers": False}

experiments["BASIC_RAG_SUPPRESS_ANSWERS"] = {"name": "BASIC_RAG_SUPPRESS_ANSWERS",
                                             "sample_ldicts_idx": SAMPLE_LDICTS_IDX,
                                             "query_prompt_idx": BASIC_RAG_PROMPT_IDX,
                                             "context_needed": True,
                                             "hyde_context_needed": False,
                                             "suppress_answers": True}

experiments["BASIC_RAG_HYDE"] = {"name": "BASIC_RAG_HYDE",
                                 "sample_ldicts_idx": SAMPLE_HYDE_LDICTS_IDX,
                                 "query_prompt_idx": BASIC_RAG_PROMPT_IDX,
                                 "context_needed": True,
                                 "hyde_context_needed": True,
                                 "suppress_answers": False}

experiments["BASIC_RAG_SEMANTIC_CHUNKING"] = {"name": "BASIC_RAG_SEMANTIC_CHUNKING",
                                              "sample_ldicts_idx": SAMPLE_SEMANTIC_LDICTS_IDX,
                                              "query_prompt_idx": BASIC_RAG_PROMPT_IDX,
                                              "context_needed": True,
                                              "hyde_context_needed": False,
                                              "suppress_answers": False}

experiments["FULL_RAG"] = {"name": "FULL_RAG",
                           "sample_ldicts_idx": SAMPLE_SEMANTIC_LDICTS_IDX, # SAMPLE_HYDE_LDICTS_IDX,
                           "query_prompt_idx": BASIC_RAG_DONT_LIE_PROMPT_IDX,
                           "context_needed": True,
                           "hyde_context_needed": False,
                           "suppress_answers": False}
