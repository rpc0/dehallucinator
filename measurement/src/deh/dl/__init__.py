from deh.assessment import QASet
from typing import List
from pathlib import Path
import os
import pandas as pd


class AssessmentDataDownloader:
    """Abstract base class to contain logic across all DataDownloaders."""

    _contexts_df: pd.DataFrame
    _qaset: List[QASet]

    def save_contexts(self, dir_path: str):
        """Save context files to directory path.
        - dir_path: directory to save context files
        """

        # Create Folder if does not exist:
        if not os.path.exists(dir_path):
            print(f"Warning: Path {dir_path} does not exist and will be created.")
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        for index, row in self._contexts_df.iterrows():
            context_id = row["context_id"]
            context = row["context"]
            with open(
                f"{dir_path}/context_{context_id}.context", "w", encoding="utf-8"
            ) as f:
                f.write(context)

    def save_question_answers(self, file_path: str):
        """Save qaset file to diretory path.
        - file_path: file to save question-answer sets
        """

        # Create Folder if does not exist:
        p = Path(file_path)
        if not os.path.exists(p.parent):
            print(f"Warning: Path {p.parent} does not exist and will be created.")

        Path(p.parent).mkdir(parents=True, exist_ok=True)

        with open(f"{file_path}", "w", encoding="utf-8") as f:
            for qas in self._qaset:
                f.write(
                    f"{qas.question}\t{qas.ground_truth}\t{qas.is_impossible}\t{qas.ref_context_id}\n"
                )
