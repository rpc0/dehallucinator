"""

This module is at an experimental stage and work on it needs to continue.
The aim is to run the RAGAS evaluate() method to obtain RAGAS metrics,
such as faithfulness, answer_relevancy, context_recall and context_precision.
The method can be run on the output of the deh experiments, which contain
the question, the context, the ground truth answer(s) and the answer generated
by the LLM.

"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    # context_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from deh_llm import get_llm
from deh_prompts import query_prompts


# =========================================================================
def get_ground_truths(qids_l, squad_raw):
    ground_truths_l = []

    # TODO: Handle multiple plausible answers
    for qid in qids_l:
        
        squad_raw_qid = squad_raw[squad_raw['qid'] == qid]
        if squad_raw_qid["is_impossible"].values[0]:
            ground_truths_l.append("")
        else:
            ground_truths_l.append(squad_raw_qid['answer'].values[0])
        
    squad_raw_qid = squad_raw[squad_raw['qid'] == qids_l[0]]

    return ground_truths_l


# =========================================================================
def create_ragas_dataset(df, experiment_name, results_folder_name):

    answer_col_name = f"{experiment_name.lower()}_llm_answer"

    df_ragas_input = df[["qid", "question", "question_context", answer_col_name]]

    df_ragas_input = pd.DataFrame(df["qid"].values, columns=["qid"])   
    df_ragas_input["answer"] = df[answer_col_name].values
    df_ragas_input["question"] = df["question"].values
    df_ragas_input["context"] = df["question_context"].values

    # df_ragas_input.drop(columns=[answer_col_name], inplace=True)
    # df_ragas_input["context"] = df_ragas_input["question_context"]
    # df_ragas_input.drop(columns=["question_context"], inplace=True)

    ground_truths_l = get_ground_truths(df_ragas_input["qid"].to_list())
    df_ragas_input["ground_truth"] = ground_truths_l
    df_ragas_input.to_csv(f'{results_folder_name}/df_ragas_input.csv', index=False)

    # df_ragas_input = pd.read_csv(f'{results_folder_name}/df_ragas_input.csv')
    ragas_data_dict = {}
    q_l = df_ragas_input["question"].to_list()
    ragas_data_dict["question"] = q_l
    a_l = df_ragas_input["answer"].to_list()
    ragas_data_dict["answer"] = a_l
    c_l = df_ragas_input["context"].to_list()
    ragas_data_dict["contexts"] = c_l
    # ragas_data_dict["ground_truth"] = [[gt] for gt in ground_truths_l]
    ragas_data_dict["ground_truth"] = ground_truths_l

    contexts_l = []
    for context in ragas_data_dict["contexts"]:
        contexts_l.append(context.split("\n\n"))
    ragas_data_dict["contexts"] = contexts_l
    # print(f"ragas_data_dict: {ragas_data_dict}")

    ragas_dataset = Dataset.from_dict(ragas_data_dict)
    # print(f"ragas_dataset: {ragas_dataset}")
    return ragas_dataset


# =========================================================================
def generate_ragas_metrics(df, experiment_name, ragas_dataset, query_prompt_idx,
                           results_folder_name):
    ragas_dataset = create_ragas_dataset(df, experiment_name, results_folder_name)

    # print(f"ragas_dataset: {ragas_dataset}")
    # for feat in ragas_dataset.features:
    #     print(feat)

    # for ele in ragas_dataset:
    #     print(ele)

    current_query_prompt = query_prompts[query_prompt_idx]
    llm = get_llm(current_query_prompt)

    # ollama_embedding_model = "avr/sfr-embedding-mistral"
    # embeddings = OllamaEmbeddings(model=ollama_embedding_model)

    embed_model_name = "BAAI/bge-base-en-v1.5"
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # system_prompt = """
    # You are a Q&A assistant. Your goal is to answer questions as accurately and concisely 
    # as possible based on the instructions and context provided. When you provide an answer, 
    # cite your source document names.
    # """

    # query_wrapper_prompt = SimpleInputPrompt("{query_str}")
    
    # #llm = OllamaLLM(llm)
    # llm = Ollama(
    #     model="qwen2.5:7b", 
    #     base_url="http://localhost:11434", 
    #     context_window=4096,
    #     system_prompt=system_prompt, 
    #     query_wrapper_prompt=query_wrapper_prompt, 
    #     temperature=0
    # )

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=llm,
        embeddings=embed_model
    )

    return result
