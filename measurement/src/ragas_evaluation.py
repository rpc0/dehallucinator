from dotenv import load_dotenv
from ragas.metrics import answer_correctness
from deh.assessment import QASetRetriever
from deh.assessment import ExperimentSet
from ragas import evaluate

load_dotenv()

qa_set = QASetRetriever.get_qasets("data/qas.tsv", sample_size=5)

experiments = []
for qa in qa_set:
    experiments.append(ExperimentSet(qa, "This is my generated answers."))

ds_exp = ExperimentSet.to_DataSet(experiments)

result = evaluate(ds_exp, metrics=[answer_correctness], is_async=False, raise_exceptions=False)


print(result)
