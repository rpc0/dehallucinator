"""

Contains functionality needed for creating Hyde articles and contexts based
on these articles. Since this process is very slow, not all Hyde-based
contexts have been created for the purposes of the experiments.

To create the Hyde-based contexts, the corresponding function is called from
the command line.

"""


# ==========================================================================
import csv
import argparse
import sys
from deh_squad_data import load_squad_data
from deh_llm import get_llm
from deh_vector_store import get_vector_store

import random

from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate


# ==========================================================================
def parse_args():
    """
    Reads in command line arguments for further processing.
    """

    parser = argparse.ArgumentParser('Script for generating hyde-based contexts for the SQuAD dataset.')

    # Optional parameter
    parser.add_argument('--sample_size', '-s', type=int, default=50,
                        help='Set the sample size for generating hyde-based contexts.')

    if len(sys.argv) == 1:
        parser.print_help()

    return parser.parse_args()


# ==========================================================================
def persist_hyde_based_contexts(csv_file_path, hyde_baased_contexts):

    # Write the the Hyde contexts to a CSV file
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["qid", "question", "hyde_article", "hyde_based_context"])
        writer.writeheader()   # Write the header row
        writer.writerows(hyde_baased_contexts)  # Write the data rows

    print(f"Data successfully written to {csv_file_path}")


# ==========================================================================
def get_hyde_based_contexts(hyde_based_contexts_path):

    hyde_based_contexts = []
    with open(hyde_based_contexts_path, mode='r') as file:
        csv_reader = csv.reader(file)

        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            hyde_based_contexts.append({"qid": row[0], "question": row[1],
                                        "hyde_article": row[2], "hyde_based_context": row[3]})

    questions_already_processed = [hbc["question"] for hbc in hyde_based_contexts]

    return hyde_based_contexts, questions_already_processed


# ==========================================================================
# Names of datasets : unique_questions, unique_qas
# Generate HYDE_SAMPLE_SIZE random samples from the unique_qas and 
# generate the hyde_based_contexts for each of the samples
def generate_hyde_based_contexts(hyde_sample_size=150):

    DATA_ROOT = "/home/spiro/Studium/Harvard/00_Capstone/deh_data_results/data"  # Set to your own data folder
    HYDE_BASED_CONTEXTS_ROOT = F"{DATA_ROOT}/hyde_based_contexts"      # Set to your own hyde-based contexts folder
    hyde_based_contexts_path = f"{HYDE_BASED_CONTEXTS_ROOT}/hyde_based_contexts.csv"

    _, _, _, unique_qas, _ = load_squad_data()
    sample = random.sample(unique_qas, hyde_sample_size)

    hyde_based_contexts_path = f"{HYDE_BASED_CONTEXTS_ROOT}/hyde_based_contexts.csv"
    hyde_based_contexts, questions_already_processed = get_hyde_based_contexts(hyde_based_contexts_path)

    hyde_article_prompt = """
        Imagine you are an expert writing a detailed explanation on the topic: '{question}'
        Your response should be comprehensive and include all key points
        that would be found in the top search result.
    """
    current_query_prompt = PromptTemplate(
        template=hyde_article_prompt,
        input_variables=["question"]
    )
    llm = get_llm(current_query_prompt)
    runnable_chain = RunnableSequence(current_query_prompt | llm)

    vector_store = get_vector_store("deh_rag", DEFAULT_CHUNKING_METHOD)

    for i, qa in enumerate(sample):
        print(f"Processing question {i}...")

        question = qa["question"]
        if question in questions_already_processed:
            print(f"Question {question} already processed. Skipping...")
            continue

        new_hyde_based_context = {}
        question = qa["question"]
        qid = qa["qid"]

        # generate Hyde article
        response = runnable_chain.invoke({"question": question})
        hyde_based_article = response.content

        # generate Hyde context based on the Hyde article
        # ?? TODO: Fix the generation of the hyde_based_contexts; the prompt is not correct
        top_docs = vector_store.similarity_search(
            query=question,
            k=5
        )
        hyde_based_context = " ".join([top_doc.page_content for top_doc in top_docs])

        new_hyde_based_context["qid"] = qid # squad_scoring.get_qid_from_question(hcb["question"], dataset)
        new_hyde_based_context["question"] = question
        new_hyde_based_context["hyde_article"] = hyde_based_article
        new_hyde_based_context["hyde_based_context"] = hyde_based_context
        
        hyde_based_contexts.append(new_hyde_based_context)

    hyde_based_contexts_persist_path = f"{HYDE_BASED_CONTEXTS_ROOT}/hyde_based_contexts.csv"
    persist_hyde_based_contexts(hyde_based_contexts_persist_path, hyde_based_contexts)


# =================================================================================================
if __name__ == '__main__':

    # Read command line parameters
    cl_args = parse_args()

    if cl_args.sample_size:
        sample_size = cl_args.sample_size
        sample_size = int(sample_size)
        generate_hyde_based_contexts(sample_size)
    else:
        generate_hyde_based_contexts()
