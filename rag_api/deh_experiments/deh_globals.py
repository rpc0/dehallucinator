""" 

Contains global variables that are used for the experiments.

"""

import sys
import os
sys.path.append(os.path.abspath("../../../deh/src"))
from squad_scoring import load_dataset

import pandas as pd
import deh_experiments_config

# Folders for storing data and the results
DATA_ROOT = "../../../deh_data_results/data"         # Set to your own data folder
RESULTS_ROOT = "../../../deh_data_results/results"   # Set to your own results folder
HYDE_BASED_CONTEXTS_ROOT = F"{DATA_ROOT}/hyde_based_contexts"   # Set to your own Hyde-based contexts folder

# SQuAD dataset
data_file = f"{DATA_ROOT}/dev-v2.0.json"
dataset = load_dataset(data_file)

# Vector Store Parameters
VECTOR_STORE_TOP_K_L = [2, 3, 4, 5, 6, 8, 12]
VECTOR_STORE_TOP_K = VECTOR_STORE_TOP_K_L[3]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
DEFAULT_CHROMA_PREFIX = "deh_rag"
DEFAULT_CHUNKING_METHOD = "naive"
DEFAULT_SEMANTIC_CHUNKING_METHOD = "naive"

CHUNK_SQUAD_DATASET = False        # Set to True to vectorize the squad dataset. If False,
                                   # then the documents and their embeddings should already
                                   # exist in the vector store.

# CONTEXT Creation Parameters
REFRESH_QUESTION_CONTEXTS = False   # Set to True to create question contexts from the vector store; if False,
                                    # the question contexts are expected to already exist in a csv file.
REFRESH_HYDE_CONTEXTS = False       # Set to True to create Hyde-based contexts; if False;
                                    # the Hyde-based contexts are expected to already exist in a csv file.                                    

RESTORE_QAS_WITH_CONTEXTS = False

# Bootstrap Parameters
SAMPLE_SIZE = 200                   # number of questions to be selected from the SQuAD dataset for an experiment
BOOTSTRAPS_N = 10000                # number of bootstraps

# Timing parameter; will contain execution time related info
execution_times_l = []

# Dataframe to store the experiments
df_experiments = pd.DataFrame(deh_experiments_config.experiments).T
