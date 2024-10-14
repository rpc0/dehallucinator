"""
Classes to manage download assessment datasets.
"""

from dataclasses import dataclass

import random
from typing import List, Optional
from datasets import Dataset
import json


@dataclass
class Context:
    """Dataclass for storing context."""

    text: str


class QASet:
    """Dataclass for storing question and associated ground truth answer."""

    question: str
    ground_truth: str
    is_impossible: bool

    def __init__(self, question, ground_truth, is_impossible) -> None:
        self.question = question
        self.ground_truth = ground_truth
        self.is_impossible = is_impossible

    def __str__(self) -> str:
        return f"question: {self.question}, ground_truth: {self.ground_truth}, is_impossible: {self.is_impossible}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_json(self) -> str:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "is_impossible": self.is_impossible,
        }


class ExperimentSet(QASet):
    answer: str
    contexts: List[str]

    def __init__(self, qaset: QASet, gen_answer: str, contexts=[]):
        self.question = qaset.question
        self.ground_truth = qaset.ground_truth
        self.is_impossible = qaset.is_impossible
        self.answer = gen_answer
        self.contexts = contexts

    def __str__(self) -> str:
        ret = super().__str__()
        ret = ret + f", answer: {self.answer}"
        return ret

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def to_DataSet(experiments: List):
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for experiment in experiments:
            questions.append(experiment.question)
            answers.append(experiment.answer)
            contexts.append(experiment.contexts)
            ground_truths.append(experiment.ground_truth)

        return Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )


class QASetRetriever:

    @classmethod
    def get_qasets(
        clz, file_path: str, sample_size: Optional[int] = None
    ) -> List[QASet]:
        qa_set = []
        with open(f"{file_path}", "r") as f:
            qa_set = [line.strip().split("\t") for line in f]

        qasets = [QASet(qas[0], qas[1], False) for qas in qa_set]

        return random.sample(qasets, k=sample_size) if sample_size else qasets
