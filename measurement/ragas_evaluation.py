# TODO: This is a legacy implementation.  Should be moved to use eval libraries/notebook.

from deh.assessment import QASetRetriever
from deh.assessment import ExperimentSet
from deh import settings

from ragas import evaluate
from ragas.metrics import answer_similarity

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

import requests
import urllib.parse
import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datasets import Dataset
import math


def create_api_answer_url(question: str):
    """Returns a formatted url for API query."""
    # URL encode the question:
    question = urllib.parse.quote(question)
    return f"http://{settings.API_ANSWER_ENDPOINT}/answer?question={question}"


def create_api_config_url():
    """Returns url for API configuration info."""
    return f"http://{settings.API_ANSWER_ENDPOINT}/"


def get_config_params():
    """Query API to retrieve RAG config params."""
    response = requests.get(create_api_config_url())
    return response.json()


def generate_answer(question: str):
    """Query API to retrieve LLM generated answer."""
    response = requests.get(create_api_answer_url(question))
    return response.json()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="RAGAS QA Evaluator")
    parser.add_argument("--qas_file", default="./data/qas/squad_qas.tsv")
    parser.add_argument("--evaluation_folder", default="./data/evaluation/")
    parser.add_argument("--cache_folder", default="./data/evaluation_cache/")
    parser.add_argument("--sample_size", default=None)

    args = parser.parse_args()

    # Console print output:
    print("RAGAS QA Evaluation")
    print(f"qas_flie: {args.qas_file}")
    print(f"Using cache folder: {args.cache_folder}")
    print(f"storing evaluation results to: {args.evaluation_folder}")
    print(f"sample_size: {args.sample_size}")
    print(f"RAG API endponig: {settings.API_ANSWER_ENDPOINT}")

    # Get API Endpoint Configuration:
    response = get_config_params()
    llm_model = response["llm_model"]
    llm_prompt = response["llm_prompt"]
    embedding_model = response["embedding_model"]
    docs_loaded = response["docs_loaded"]
    print(
        f"Configuration using llm model: {llm_model} and embedding model: {embedding_model}."
    )
    print(f"{docs_loaded} documents are loaded in vector store.")

    # Check if cache exists:
    cache_file = f"{args.cache_folder}/experiment.pkl"
    cached_experiment = os.path.exists(cache_file)

    if not cached_experiment:

        # Load the QA Pairs:
        qa_set = QASetRetriever.get_qasets(args.qas_file, sample_size=args.sample_size)
        qa_set_cnt = len(qa_set)

        # Foreach Question, retrieve the LLM generated Answer:
        experiments = []
        cnt = 1
        for qa in qa_set:
            print(f"Processing {cnt} of {qa_set_cnt} question/answer pairs.")
            try:
                response = generate_answer(qa.question)
                answer = response["response"]["answer"]
                contexts = response["response"]["context"]

                experiments.append(
                    ExperimentSet(
                        qaset=qa,
                        gen_answer=answer,
                        contexts=contexts.split("------------"),
                    )
                )
            except:
                print(f"Error processing {cnt} of {qa_set_cnt}")

            cnt = cnt + 1

        exp_df = ExperimentSet.to_DataSet(experiments).to_pandas()
        exp_df["llm_model"] = llm_model
        exp_df["llm_prompt"] = llm_prompt
        exp_df["embedding_model"] = embedding_model
        exp_df["docs_loaded"] = docs_loaded
        exp_df["assessment_llm"] = settings.ASSESSMENT_LLM_MODEL
        exp_df["assessment_embedding"] = settings.ASSESSMENT_EMBEDDING_MODEL

        # Cache results:
        if not os.path.exists(args.cache_folder):
            Path(args.cache_folder).mkdir(parents=True, exist_ok=True)

        exp_df.to_pickle(cache_file)

    # Load cached results:
    exp_df = pd.read_pickle(cache_file)

    # Models used for assessment:
    embedding = OllamaEmbeddings(
        base_url=settings.OLLAMA_URL,
        model=settings.ASSESSMENT_EMBEDDING_MODEL,
    )

    llm = Ollama(
        base_url=settings.OLLAMA_URL,
        model=settings.ASSESSMENT_LLM_MODEL,
    )

    # TODO: Make this a configuration parameter:
    chunk_size = 500
    chunked_df = np.array_split(exp_df, math.ceil(len(exp_df) / chunk_size))
    results = []

    for chunk_df in chunked_df:
        ds_exp = Dataset.from_pandas(chunk_df)
        result = evaluate(
            ds_exp, metrics=[answer_similarity], embeddings=embedding, llm=llm
        )

        results.append(result)
        print(f"Completed evaluation with total values of {result}.")

    # Save experiment results:
    result_df = pd.concat([result.to_pandas() for result in results])

    timestr = time.strftime("%Y%m%d-%H%M%S")
    path_to_evaluation = f"{args.evaluation_folder}/{timestr}.csv"

    filepath = Path(path_to_evaluation)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(path_to_evaluation, index=False)

    # Delete cached experiment on completion:
    os.remove(cache_file)
