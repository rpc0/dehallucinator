import sys
import os
sys.path.append(os.path.abspath("../../../deh/src"))
from squad_scoring import load_dataset

import pandas as pd
import deh_experiments_config

# # Folders for storing data and the results
DATA_ROOT = "../../../deh_data_results/data"         # Set to your own data folder
RESULTS_ROOT = "../../../deh_data_results/results"   # Set to your own results folder
HYDE_BASED_CONTEXTS_ROOT = F"{DATA_ROOT}/hyde_based_contexts"   # Set to your own results folder

# SQuAD dataset
data_file = f"{DATA_ROOT}/dev-v2.0.json"
# dataset = squad_scoring.load_dataset(data_file)
dataset = load_dataset(data_file)

# Vector Store Parameters
VECTOR_STORE_TOP_K_L = [2, 3, 4, 5, 6, 8, 12]
VECTOR_STORE_TOP_K = VECTOR_STORE_TOP_K_L[3]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
DEFAULT_CHROMA_PREFIX = "deh_rag"
#DEFAULT_CHUNKING_METHOD = "pseudo_semantic"
DEFAULT_CHUNKING_METHOD = "per_article"
DEFAULT_SEMANTIC_CHUNKING_METHOD = "pseudo_semantic"

CHUNK_SQUAD_DATASET = False        # Set to True to vectorize the squad dataset. If False,
                                   # then the documents and their embeddings should already
                                   # exist in the vector store.

# # CONTEXT Creation Parameters
REFRESH_QUESTION_CONTEXTS = False   # Set to True to create question contexts from the vector store; 
                                    # if False, the question contexts are loaded from a csv file.
REFRESH_HYDE_CONTEXTS = False       # Set to True to create hyde contexts; if False,
#                                   # the hyde contexts are loaded from a csv file.                                    

RESTORE_QAS_WITH_CONTEXTS = False

# # Bootstrap Parameters
SAMPLE_SIZE = 10
BOOTSTRAPS_N = 1000

# Timing
execution_times_l = []

# Dataframe to store the experiments
df_experiments = pd.DataFrame(deh_experiments_config.experiments).T

# #TODO check if setting the seed makes sense
# # SEED = 42
# # set_seed = random.seed(SEED)

# PERSIST_ANSWER_SAMPLES = False   # Set to True to persist the llm answers for each sample, for each experiment