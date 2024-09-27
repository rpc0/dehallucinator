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
from pathlib import Path


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
    parser.add_argument("--sample_size", type=int, default=10)

    args = parser.parse_args()

    # Console print output:
    print("RAGAS QA Evaluation")
    print(f"qas_flie: {args.qas_file}")
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

    # Load the QA Pairs:
    qa_set = QASetRetriever.get_qasets(args.qas_file, sample_size=args.sample_size)
    qa_set_cnt = len(qa_set)

    # Foreach Question, retrieve the LLM generated Answer:
    experiments = []
    cnt = 1
    for qa in qa_set:
        print(f"Processing {cnt} of {qa_set_cnt} question/answer pairs.")
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

        cnt = cnt + 1

    ds_exp = ExperimentSet.to_DataSet(experiments)

    # Models used for assessment:
    embedding = OllamaEmbeddings(
        base_url=settings.OLLAMA_URL,
        model=settings.ASSESSMENT_EMBEDDING_MODEL,
    )

    llm = Ollama(
        base_url=settings.OLLAMA_URL,
        model=settings.ASSESSMENT_LLM_MODEL,
    )

    result = evaluate(
        ds_exp, metrics=[answer_similarity], embeddings=embedding, llm=llm
    )

    print(f"Completed evaluation with total values of {result}.")

    result_df = result.to_pandas()
    result_df["llm_model"] = llm_model
    result_df["llm_prompt"] = llm_prompt
    result_df["embedding_model"] = embedding_model
    result_df["docs_loaded"] = docs_loaded
    result_df["assessment_llm"] = settings.ASSESSMENT_LLM_MODEL
    result_df["assessment_embedding"] = settings.ASSESSMENT_EMBEDDING_MODEL

    timestr = time.strftime("%Y%m%d-%H%M%S")
    path_to_evaluation = f"{args.evaluation_folder}/{timestr}.csv"

    filepath = Path(path_to_evaluation)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(path_to_evaluation, index=False)