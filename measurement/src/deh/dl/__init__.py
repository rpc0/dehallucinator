from deh.assessment import QASet, Context
from typing import List


class AssessmentDataDownloader:
    """Abstract base class to contain logic across all DataDownloaders."""

    _contexts: List[Context]
    _qaset: List[QASet]

    def save_contexts(self, dir_path: str):
        """Save context files to directory path.
        - dir_path: directory to save context files
        """
        context_count = 1
        for context in self._contexts:
            with open(f"{dir_path}context_{context_count}.context", "w") as f:
                f.write(context.text)
                context_count = context_count + 1

    def save_question_answers(self, file_path: str):
        """Save qaset file to diretory path.
        - file_path: file to save question-answer sets
        """
        with open(f"{file_path}", "w") as f:
            for qas in self._qaset:
                f.write(f"{qas.question}\t{qas.ground_truth}\n")
