
import csv
import json

# from ..squad_scoring import load_dataset

DATA_ROOT = "/home/spiro/Studium/Harvard/00_Capstone/deh_data_results/data"  # Set to your own data folder
# RESULTS_ROOT = "../../../deh_data_results/results"   # Set to your own results folder
# HYDE_BASED_CONTEXTS_ROOT = F"{DATA_ROOT}/hyde_based_contexts"   # Set to your own hyde-based contexts folder


# =================================================================================================
def load_dataset(data_file):
    """
    Loads the data file from the file specified in "data_file" and returns a list. Datafile has
    to be a json file that corresponds to the format of the dev set found on
    https://rajpurkar.github.io/SQuAD-explorer/

    Args:

      data_file (string): the path to the data file

    Returns:

      dataset (list): list of the complete dataset; each entry contains data for one single article
                      (e.g. Harvard university), including title, context and qas
    """

    dataset = []

    # Load the file that contains the dataset (expected to be in json format)
    try:
        with open(data_file) as f:
            # dataset_json: dict with 'version' and 'data' as keys
            # 'data' contains the real data (see next variable)
            dataset_json = json.load(f)       
            # list of articles; each entry contains data for one single article
            # (e.g. Harvard university), including title, context and qas
            dataset = dataset_json['data']    
    except FileNotFoundError:
        print(f"Error: the data file '{data_file}' could not be found...")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: the data file '{data_file}' could not be read, since it is not a valid JSON file...")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        exit(1)

    return dataset


# ==========================================================================
def load_squad_data():

    data_file = f"{DATA_ROOT}/dev-v2.0.json"
    dataset = load_dataset(data_file)

    articles = []
    contexts = []
    qas = []

    for article in dataset:
        title = article["title"]
        articles.append(title)
        for p in article['paragraphs']:
            context = p["context"]
            contexts.append(context)
            for qa in p['qas']:
                question = qa["question"]
                id = qa["id"]
                is_impossible = qa["is_impossible"]
                if is_impossible:
                    for pa in qa["plausible_answers"]:
                        answer = pa["text"]
                        qas.append({"title": title, "context": context, "qid": id, "question": question, 
                                    "is_impossible": is_impossible, "answer": answer})
                else:
                    for a in qa["answers"]:
                        answer = a["text"]
                        qas.append({"title": title, "context": context, "qid": id, "question": question, 
                                    "is_impossible": is_impossible, "answer": answer})
                        
    # Store dataset as a csv file
    csv_file = f"{DATA_ROOT}/dev-v2.0.csv"
    with open(csv_file, mode='w') as file:
        fieldnames = ['title', 'context', 'qid', 'question', 'is_impossible', 'answer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for qa in qas:
            writer.writerow(qa)

    unique_questions = list(set([qa["question"] for qa in qas]))
    unique_qas = [dict(t) for t in {tuple(d.items()) for d in qas}]

    print(f"#articles in the dataset:     {len(articles)}")
    print(f"#contexts in the dataset:   {len(contexts)}")
    print(f"#questions in the dataset: {len(qas)}")   
    print(f"#unique questions in the dataset: {len(unique_questions)}")
    print(f"#unique qas in the dataset: {len(unique_qas)}")

    # TODO: unique_qas is probably no really unique, since it is based on the dictionary
    #       representation of the qas, which might contain several slightly different answers
    #       for the same question
    return articles, contexts, qas, unique_qas, unique_questions
