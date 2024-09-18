from deh.assessment import QASetRetriever
from deh.assessment import ExperimentSet

from ragas import evaluate
from ragas.metrics import answer_similarity

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

import requests
import urllib.parse


def generate_answer(question: str):
    question = urllib.parse.quote(question)
    response = requests.get("http://localhost/api/answer?question=" + question)
    return response


qa_set = QASetRetriever.get_qasets("data/qas/squad_qas.tsv", sample_size=10)

experiments = []
for qa in qa_set:
    experiments.append(
        ExperimentSet(
            qaset=qa,
            gen_answer=generate_answer(qa.question),
            contexts=["context1", "context2"],
        )
    )

ds_exp = ExperimentSet.to_DataSet(experiments)

embedding = OllamaEmbeddings(
    base_url="http://localhost:7869",
    model="all-minilm:latest",
)

llm = Ollama(
    base_url="http://localhost:7869",
    model="tinyllama:latest",
)

# TODO: Add LLM-model & embedding model & vector store hyper parameters capture from chain.
result = evaluate(ds_exp, metrics=[answer_similarity], embeddings=embedding, llm=llm)

print(result)
