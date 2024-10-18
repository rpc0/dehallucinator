import pandas as pd
import requests
import urllib.parse as parse

from typing import List
from deh.assessment import QASet

from deh import settings


def get_config_params():
    """Query API to retrieve RAG config params."""
    response = requests.get(f"http://{settings.API_ANSWER_ENDPOINT}/")
    return response.json()


def generate_api_response(api_endpoint, **kwargs):
    """Query API to retrieve LLM generated answer."""

    # URLencode potential param values:
    kwargs = dict([(key, parse.quote(kwargs[key])) for key in kwargs])

    response = requests.get(api_endpoint(**kwargs))
    return response.json()


def generate_experiment_dataset(qa_set: List[QASet], fxn_parse_results, api_endpoint):
    """Execute API end-point calls and combine in DataFrame"""

    # Foreach Question, retrieve the LLM generated Answer:
    exp_df: pd.DataFrame = None
    qa_set_cnt = len(qa_set)
    cnt = 1
    for qa in qa_set:
        print(f"Processing {cnt} of {qa_set_cnt} question/answer pairs.")
        try:
            response = generate_api_response(api_endpoint, q=qa.question)

            # Add qa reference measures:
            response["reference"] = qa.to_json()

            df = fxn_parse_results(response)

            # Generate a "group-by" reference id
            df["reference_id"] = cnt

            if exp_df is None:
                exp_df = df
            else:
                exp_df = pd.concat([exp_df, df])

        except Exception as exc:
            print(f"Error processing {cnt} of {qa_set_cnt}")
            print(exc)

        cnt = cnt + 1
    return exp_df


def add_setting_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    """Add standard meta-data columns to a Dataframe."""

    response = get_config_params()

    llm_model = response["llm_model"]
    llm_prompt = response["llm_prompt"]
    embedding_model = response["embedding_model"]
    docs_loaded = response["docs_loaded"]

    input_df["llm_model"] = llm_model
    input_df["llm_prompt"] = llm_prompt
    input_df["embedding_model"] = embedding_model
    input_df["docs_loaded"] = docs_loaded
    input_df["assessment_llm"] = settings.ASSESSMENT_LLM_MODEL
    input_df["assessment_embedding"] = settings.ASSESSMENT_EMBEDDING_MODEL

    return input_df
