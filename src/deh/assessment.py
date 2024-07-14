'''
Classes to manage download assessment datasets.
'''

from abc import ABC, abstractmethod
import requests
from urllib.parse import urlparse
from dataclasses import dataclass
import os
import jmespath
import json
import random
from typing import List
from datasets import Dataset


@dataclass
class Context:
    '''Dataclass for storing context.'''

    text: str


class QASet:
    '''Dataclass for storing question and associated ground truth answer.'''

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


class ExperimentSet(QASet):
    answer: str
    contexts: List[str]

    def __init__(self, qaset: QASet, gen_answer: str):
        self.question = qaset.question
        self.ground_truth = qaset.ground_truth
        self.is_impossible = qaset.is_impossible
        self.answer = gen_answer
        self.contexts = []

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
                # "contexts" : contexts,
                "ground_truth": ground_truths,
            }
        )


class AssessmentDataDownloader:
    '''Abstract base class to contain logic across all DataDownloaders.'''

    _contexts: List[Context]
    _qaset: List[QASet]

    def save_contexts(self, dir_path: str):
        '''Save context files to directory path.
        - dir_path: directory to save context files
        '''
        context_count = 1
        for context in self._contexts:
            with open(f"{dir_path}context_{context_count}.context", "w") as f:
                f.write(context.text)
                context_count = context_count + 1

    def save_question_answers(self, file_path: str):
        '''Save qaset file to diretory path.
        - file_path: file to save question-answer sets
        '''
        with open(f"{file_path}", "w") as f:
            for qas in self._qaset:
                f.write(f"{qas.question}\t{qas.ground_truth}\n")


class SquadAssessmentDataDownloader(AssessmentDataDownloader):
    '''Specific implementation of DataDownload for SQuAD2.0 dataset.'''

    DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    FULL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"

    def __init__(self, dl_url: str = DEV, filter_impossible=True, cache_dir: str = None) -> None:
        '''Download and convert file to expected format.
        - dl_url: url of the file to download [DEV|FULL]
        - filter_impossible: filter out impossible questions [True|False]
        - cache_dir: (optional) directory to store downloaded file
        '''
        url = urlparse(dl_url)
        f_name = os.path.basename(url.path)

        r = requests.get(dl_url)
        json_data = r.json()

        if cache_dir:
            with open(f'{cache_dir}{f_name}', 'w') as f:
                json.dump(json_data, f)

        # Collect all of the context documents:
        contexts = jmespath.search("data[*].paragraphs[*].context", json_data)
        self._contexts = [Context(context[0]) for context in contexts]

        # Collect question-answer pairs:
        filter_str = "?is_impossible!=True" if filter_impossible else "*"
        possible_qas = jmespath.search(f"data[*].paragraphs[*].qas[{filter_str}]", json_data)
        self._qaset = []
        for qa in possible_qas:
            qa = qa[0][0]
            question = qa["question"]
            answers = set([a["text"] for a in qa["answers"]])
            for answer in answers:
                self._qaset.append(QASet(question, answer, False))


class QASetRetriever:

    @classmethod
    def get_qasets(clz, file_path: str, sample_size=None) -> List[QASet]:
        qa_set = []
        with open(f"{file_path}", "r") as f:
            qa_set = [line.strip().split("\t") for line in f]

        qasets = [QASet(qas[0], qas[1], False) for qas in qa_set]

        return random.sample(qasets, k=sample_size) if sample_size else qasets
