import os
import json

import requests
from urllib.parse import urlparse
from pathlib import Path

from deh.assessment import QASet
from deh.dl import AssessmentDataDownloader
from typing import Optional

import pandas as pd


class SquadAssessmentDataDownloader(AssessmentDataDownloader):
    """Specific implementation of DataDownload for SQuAD2.0 dataset."""

    DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    FULL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"

    def parse_json_file(self, json_data, limit_size: Optional[int]) -> pd.DataFrame:
        """Parses the SQuAD JSON file and returns a normalized df."""

        # Normalize data-structure:
        df_table = pd.json_normalize(
            json_data["data"],
            record_path=["paragraphs", "qas", "answers"],
            meta=[
                "title",
                ["paragraphs", "context"],
                ["paragraphs", "qas", "question"],
                ["paragraphs", "qas", "is_impossible"],
            ],
        )

        # Rename and Select Key Columns:
        selector = {
            "text": "answer",
            "paragraphs.context": "context",
            "paragraphs.qas.question": "question",
            "paragraphs.qas.is_impossible": "is_impossible",
        }
        df_table = df_table.rename(columns=selector)[[*selector.values()]]

        # Handle impossible question structure:
        impossible_df = pd.json_normalize(
            json_data["data"],
            record_path=[
                "paragraphs",
                "qas",
            ],
            meta=[
                "title",
                ["paragraphs", "context"],
            ],
        )

        impossible_df["answer"] = ""
        rename = {"paragraphs.context": "context"}
        impossible_df = impossible_df.rename(columns=rename)[
            ["question", "answer", "context", "is_impossible"]
        ]

        impossible_df = impossible_df[
            impossible_df["is_impossible"] == True
        ]  # noqa: E712
        df_table = pd.concat([df_table, impossible_df], ignore_index=True)

        # Create unique context id:
        df_table["context_id"] = df_table.groupby(["context"]).ngroup()

        # De-duplicate rows:
        de_dupe_df = df_table.drop_duplicates()

        # Optional size limit:
        if limit_size:
            de_dupe_df = de_dupe_df[0:limit_size]

        return de_dupe_df

    def __init__(
        self,
        dl_url: str = DEV,
        cache_dir: str = None,
        limit_size: Optional[int] = None,
    ) -> None:
        """Download and convert file to expected format.
        - dl_url: url of the file to download [DEV|FULL]
        - filter_impossible: filter out impossible questions [True|False]
        - cache_dir: (optional) directory to store downloaded file
        """
        url = urlparse(dl_url)
        f_name = os.path.basename(url.path)

        json_data = None

        if cache_dir:
            cache_file_path = f"{cache_dir}/{f_name}"

            if os.path.exists(cache_file_path):
                # Load cached data download:
                with open(cache_file_path) as json_file:
                    json_data = json.load(json_file)

            else:
                r = requests.get(dl_url)
                json_data = r.json()

                if not os.path.exists(cache_dir):
                    print(
                        f"Warning: Path {cache_file_path} does not exist and will be created."
                    )

                Path(cache_dir).mkdir(parents=True, exist_ok=True)

                with open(cache_file_path, "w+") as f:
                    json.dump(json_data, f)

        # Normalized dataframe:
        df = self.parse_json_file(json_data, limit_size=limit_size)
        self._df = df

        # Contexts (context_id, context)
        self._contexts_df = df.groupby("context_id", as_index=False).agg(
            context=("context", "first"),
        )

        # Iterate, create qaset
        self._qaset = []
        for index, row in df.iterrows():
            question = row["question"]
            answer = row["answer"]
            possible = row["is_impossible"]
            context_id = row["context_id"]
            self._qaset.append(QASet(question, answer, possible, context_id))


def download_squad_qa_dataset(
    cache_folder: str,
    context_folder: str,
    qas_file: str,
    limit_size: Optional[int] = None,
):
    dl = SquadAssessmentDataDownloader(cache_dir=cache_folder, limit_size=limit_size)
    dl.save_contexts(context_folder)
    dl.save_question_answers(qas_file)
