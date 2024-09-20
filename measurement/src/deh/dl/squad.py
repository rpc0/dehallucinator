import os
import jmespath
import json

import requests
from urllib.parse import urlparse
from pathlib import Path

from deh.assessment import QASet
from deh.dl import AssessmentDataDownloader


class SquadAssessmentDataDownloader(AssessmentDataDownloader):
    """Specific implementation of DataDownload for SQuAD2.0 dataset."""

    DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    FULL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"

    def __init__(
        self, dl_url: str = DEV, filter_impossible=True, cache_dir: str = None
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

                if not os.path.exists(cache_file_path):
                    print(
                        f"Warning: Path {cache_file_path} does not exist and will be created."
                    )

                Path(cache_file_path).mkdir(parents=True, exist_ok=True)

                with open(cache_file_path, "w") as f:
                    json.dump(json_data, f)

        # Collect all of the context documents:
        contexts = jmespath.search("data[*].paragraphs[*].context[]", json_data)
        self._contexts = contexts

        # Collect question-answer pairs:
        filter_str = "?is_impossible!=True" if filter_impossible else "*"
        possible_qas = jmespath.search(
            f"data[*].paragraphs[*].qas[{filter_str}][]", json_data
        )
        self._qaset = []
        for qas in possible_qas:
            for qa in qas:
                question = qa["question"]
                answers = set([a["text"] for a in qa["answers"]])
                for answer in answers:
                    self._qaset.append(QASet(question, answer, False))


def download_squad_qa_dataset(cache_folder: str, context_folder: str, qas_file: str):
    dl = SquadAssessmentDataDownloader(cache_dir=cache_folder)
    dl.save_contexts(context_folder)
    dl.save_question_answers(qas_file)
