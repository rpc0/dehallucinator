"""

Adapted from the official evaluation script for SQuAD version 2.0.

Calculates the EM and F1 scores for ("preds" contains predicted answers):

    - all questions in preds
    - all questions in preds that have answers
    - all questions in preds that don't have answers

It also generates additional statistics and plots precision-recall curves if an additional
na_prob.json file is provided or if no answer probabilities are simulated (plots are stored
in "OPTS.image_dir"). This na_prob.json file is expected to map qid's to the model's predicted
probability that a question is unanswerable.

The script expects a .csv file with the predicted answers that contains at least two columns,
one of them named "question" that contains the questions in string format (i.e. not the qid's)
and the other one name "answer" that contains the predicted answer for each corresponding question.

The script also expects a .json file that contains the dataset (e.g. dev-v2.0.json) that corresponds
to the format of the dev dataset found on https://rajpurkar.github.io/SQuAD-explorer/.

HOW TO USE (from the command line):

usage: Adapted evaluation script for SQuAD version 2.0. [-h] [--out-file eval.json] [--na-prob-file na_prob.json] [-s]
                                                        [--na-prob-thresh NA_PROB_THRESH] [--out-image-dir out_images]
                                                        [--verbose]
                                                        data.json pred.json

positional arguments:

  data.json             Input data JSON file: path to the dataset file (e.g. dev-v2.0.json).
  pred.json             Model predictions: path to the predictions file (e.g. predictions.csv). The predictions file
                        has to be a csv file that contains at least two columns, one of them named "question" that
                        contains the questions in string format (i.e. not the qid's) and the other one named "answer"
                        that contains the predicted answers for each of the corresponding questions.

optional arguments:

  -h, --help            show this help message and exit
  --out-file eval.json, -o eval.json
                        Write accuracy metrics to file (default is stdout).
  --na-prob-file na_prob.json, -n na_prob.json
                        Model estimates of probability of no answer.
  -s, --na-prob-sim     If used, model estimates of probability of no answer are simulated.
  --na-prob-thresh NA_PROB_THRESH, -t NA_PROB_THRESH
                        Predict "" if no-answer probability exceeds this (default = 1.0).
  --out-image-dir out_images, -p out_images
                        Save precision-recall curves to directory.
  --verbose, -v
usage: Adapted evaluation script for SQuAD version 2.0. [-h] [--out-file eval.json] [--na-prob-file na_prob.json] [-s]
                                                        [--na-prob-thresh NA_PROB_THRESH] [--out-image-dir out_images] [--verbose]
                                                        data.json pred.json

Note that you you can also directly call the function "calc_squad_metrics", which calculates the metrics. It takes
the dataset, the predictions and the (optional) no answer probabilities as arguments and returns a dictionary with
the metrics.

Args of calc_squad_metrics:

      dataset (list):       list of articles; each entry contains data for one single article 
                            (e.g. Harvard university), including title, contexts and qas
                            (typically, read from a json file, such "dev-v2.0.json" file)

      preds (dictionary):   dictionary that holds predicitons to be evaluated, one answer
                            per question (id)

      na_probs (dictionary): per question (id), the probability that the question is unanswerable
                            (such as assessed by the model that did the predictions)

"""

import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys

import matplotlib
import matplotlib.pyplot as plt

from csv import DictReader # , writer

# Global variable for command line parameters
OPTS = None


# =================================================================================================
def parse_args():
    """
    Reads in command line arguments for further processing.
    """

    parser = argparse.ArgumentParser('Adapted evaluation script for SQuAD version 2.0.')

    # The next two arguments are required, since they are positional...
    parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
    parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')

    # ... and these are optional
    parser.add_argument('--out-file', '-o', metavar='eval.json', 
                        help='Write accuracy metrics to file (default is stdout).')

    parser.add_argument('--na-prob-file', '-n', metavar='na_prob.json',
                        help='Model estimates of probability of no answer.')

    parser.add_argument('-s', '--na-prob-sim', action="store_true",
                        help='If used, model estimates of probability of no answer are simulated.')

    parser.add_argument('--na-prob-thresh', '-t', type=float, default=1.0,
                        help='Predict "" if no-answer probability exceeds this (default = 1.0).')

    parser.add_argument('--out-image-dir', '-p', metavar='out_images', default=None,
                        help='Save precision-recall curves to directory.')

    parser.add_argument('--verbose', '-v', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()

    return parser.parse_args()


# =================================================================================================
def make_qid_to_has_ans(dataset):
    """
    Returns a dictionary, with for each question (identified by its id) a bool that indicates whether 
    or not the question has an answer.

    Args:
      dataset (list): list of articles (each one containing paragraphs, each paragraph containing questions and
                      answers, etc.; i.e. the complete dataset)

    Returns:
      dictionary: dictionary with qid's specifying for each of them, if they have an answer or not (True/False)
    """

    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])

    return qid_to_has_ans


# =================================================================================================
def normalize_answer(s):
    """
    Normalizes a string (answer), i.e. transforms to lower text and removes punctuation,
    articles and extra whitespace.

    Args:

      s (string): the answer to be normalized

    Returns:

      string: the normalized answer
    """

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# =================================================================================================
def get_tokens(s):
    """
    Gets the tokens for string s
    """

    if not s:
        return []
    return normalize_answer(s).split()


# =================================================================================================
def compute_exact(a_gold, a_pred):
    """
    Compares the gold answer to the predicted answer

    Args:

      a_gold (string): string that contains the gold answer
      a_pred (string): string that contains the predicrted answer

    Returns:

      int: 1 if both are equal, 0 else
    """

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


# =================================================================================================
def compute_f1(a_gold, a_pred):
    """
    Computes the f1 score, based on the gold answer and the predicted answer. The computation uses
    precision and recall. Here is an explanation of both terms:

    Precision measures the proportion of correctly predicted tokens (or words) from the total tokens
    predicted as part of the answer. In other words:

    Precision = (number of correctly predicted tokens) / (number of total tokens predicted)

    Recall measures the proportion of correctly predicted tokens (or words) out of the total tokens
    that are part of the gold answer. In other words:

    Recall = (number of correctly predicted tokens) / (number of total tokens in the gold answer)

    The F1 score is the harmonic mean of precision and recall. It is calculated as follows:

    F1 = (2 * precision * recall) / (precision + recall)

    The function 

    Args:

        a_gold (string): string that contains the gold answer
        a_pred (string): string that contains the predicrted answer

    Returns:

        dicionary (float, float, float):
            - precision: the precision score
            - recall: the recall score
            - f1 score: the formula is (2 * precision * recall) / (precision + recall))
                        (the harmonic mean of precision and recall)

    """

    # Get the tokens for the gold answer and the predicted answer
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    # Get the common tokens and the number of common tokens.
    # When words occur multiple times in the prediction or ground truth, their contributions
    # to precision and recall are based on the minimum number of occurrences in both sets (=overlap).
    # To get the overlap, the & operator is used, whereby the resulting dictionary contains the
    # minimum number of occurrences for each common token. The sum of these values is the number
    # of correctly predicted tokens.
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    correctly_predicted_tokens_cnt = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        # TODO --> check if this is correct
        agree = (gold_toks == pred_toks)
        return {"precision": int(agree), "recall": int(agree), "f1": int(agree)}
        #return int(gold_toks == pred_toks)
    if correctly_predicted_tokens_cnt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Calculate the f1 score
    precision = 1.0 * correctly_predicted_tokens_cnt / len(pred_toks)
    recall = 1.0 * correctly_predicted_tokens_cnt / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    scores_dict = {"precision": precision, "recall": recall, "f1": f1}
    return scores_dict


# =================================================================================================
def get_raw_scores(dataset, preds):
    """
    Gets exact scores and F1 scores per question based on the dataset and the predictions

    Args:

      dataset (list): list of articles (each one containing paragraphs, each paragraph containing questions and answers)
      preds (dictionary): the predictions to be evaluated; dictionary contains one entry per question id with either an answer 
                          if one is predicted, or an empty string

    Returns:

      exact_scores (dictionary): for each question, the exact score (either 0 or 1)
      precision_scores (dictionary): for each question, the precision score (a value between 0 and 1)
      recall_scores (dictionary): for each question, the recall score (a value between 0 and 1)      
      f1_scores (dictionary): for each question, the f1 score (a value between 0 and 1)

    """

    exact_scores = {}
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:

                qid = qa['id']  # get id of current question

                # get the gold answers for current question (if question is answerable)
                gold_answers = [a['text'] for a in qa['answers'] if normalize_answer(a['text'])] 

                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']

                # If there is no prediction in the predictions dict for the current question
                # print a message
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue

                # Get the predicted answer
                a_pred = preds[qid]

                # Take max over all gold answers for exact scores and f1 scores alike
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)

                # Get precision, recall and f1 scores for the current question
                # TODO: refactor, since for loop is executed three times!
                precision_scores[qid] = max(compute_f1(a, a_pred)["precision"] for a in gold_answers)
                recall_scores[qid] = max(compute_f1(a, a_pred)["recall"] for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred)["f1"] for a in gold_answers)

    return exact_scores, precision_scores, recall_scores, f1_scores


# =================================================================================================
def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    """
    Recalculates scores based on the no answer probabilities generated by the prediction model (scores may
    be EM or F1 scores). Works as follows:

      - per question, check if model-generated probability for no answer is above threshold
      - if yes, model is deemed as having predicted no answer; if no, then model predicts that there is an answer
      - if yes --> if question is really unanswerable, assign a score of 1.0, else assign 0.0
      - if no --> keep the current score as it is

    Args:

      scores (dictionary):          contains scores (either exact or f1), with one entry per question (id)
      na_probs (dictionary):        no answer probability for each question (id) (generated by the model)
      qid_to_has_ans (dictionary):  per question (id), contains True if answerable, False if not
      na_prob_threshold (float):    threshold above which a no answer probability is interpreted 
                                    as a prediction of no answer

    Returns:

    new°scores (dictionary):       updated scoring dictionary with scores updated based on threshold
    """

    new_scores = {}

    for qid, s in scores.items():
        # predicted probability for no answer above the threshold?
        pred_na = na_probs[qid] > na_prob_thresh

        if pred_na:  # above threshold
            # if no answer predicted and question is unanswerable, assign score of 1.0, else 0.0
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:        # not above threshold
            new_scores[qid] = s  # keep current score value

    return new_scores


# =================================================================================================
def make_eval_dict(exact_scores, precision_scores, recall_scores, f1_scores, qid_list=None):
    """
    Constructs the dictionary that holds the EM, preciasion, reacall and F1 scores for the complete predictions data set
    # TODO: change example to include precision and recall scores as well
    example output of the function: OrderedDict({'exact': 64.81091552261434, 'f1': 67.60971132981268, 'total': 11873})

    Args:

      exact_scores (dictionary)     : contains the exact scores, one per question (id)
      precision_scores (dictionary) : contains the precision scores, one per question (id)
      recall_scores (dictionary)    : contains the recall scores, one per question (id)
      f1_scores (dictionary)        : contains the f1 scores, one per question (id)
      qid_list (list)               : list of question ids of the questions to be evaluated

    Returns:

      dictionary: dict that holds the final EM, precision, recall and F1 scores for the whole predictions
                  dataset and the total number of questions. EM, precision, recall and F1 scores are averages
                  over all corresponding scores in the dictionaries.
    """

    if not qid_list: # compute scores for all questions
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('precision', 100.0 * sum(precision_scores.values()) / total),
            ('recall', 100.0 * sum(recall_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else: # compute scores for a subset of questions
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('precision', 100.0 * sum(precision_scores[k] for k in qid_list) / total),
            ('recall', 100.0 * sum(recall_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


# =================================================================================================
def merge_eval(main_eval, new_eval, prefix):
    """
    Merges the new_eval dictionary into the main_eval dictionary

    Args:

      main_eval (dictionary): the main dictionary, that contains the current evaluation metrics,
                              calculated up to now

      new_eval (dictionary): the dictionary that contains new evaluation metrics that is to be merged
                            into main_eval

    Returns:

      None
    """

    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


# =================================================================================================
def plot_pr_curve(precisions, recalls, out_image, title):
    """
    #TODO --> comment this function
    """

    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.savefig(out_image)
    plt.clf()


# =================================================================================================
def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans, out_image=None, title=None):
    """
    #TODO --> comment this function
    """

    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    true_pos = 0.0
    cur_p = 1.0
    cur_r = 0.0
    precisions = [1.0]
    recalls = [0.0]
    avg_prec = 0.0
    for i, qid in enumerate(qid_list):
        if qid_to_has_ans[qid]:
            true_pos += scores[qid]
        cur_p = true_pos / float(i+1)
        cur_r = true_pos / float(num_true_pos)
        if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i+1]]:
            # i.e., if we can put a threshold after this point
            avg_prec += cur_p * (cur_r - recalls[-1])
            precisions.append(cur_p)
            recalls.append(cur_r)
    if out_image:
        plot_pr_curve(precisions, recalls, out_image, title)
    return {'ap': 100.0 * avg_prec}


# =================================================================================================
def run_precision_recall_analysis(main_eval, exact_raw, f1_raw, na_probs, 
                                  qid_to_has_ans, out_image_dir):
    """
    #TODO --> comment this function
    """

    if out_image_dir and not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
    if num_true_pos == 0:
        return
    pr_exact = make_precision_recall_eval(
        exact_raw, na_probs, num_true_pos, qid_to_has_ans,
        out_image=os.path.join(out_image_dir, 'pr_exact.png'),
        title='Precision-Recall curve for Exact Match score')
    pr_f1 = make_precision_recall_eval(
        f1_raw, na_probs, num_true_pos, qid_to_has_ans,
        out_image=os.path.join(out_image_dir, 'pr_f1.png'),
        title='Precision-Recall curve for F1 score')
    oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
    pr_oracle = make_precision_recall_eval(
        oracle_scores, na_probs, num_true_pos, qid_to_has_ans,
        out_image=os.path.join(out_image_dir, 'pr_oracle.png'),
        title='Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)')

    merge_eval(main_eval, pr_exact, 'pr_exact')
    merge_eval(main_eval, pr_f1, 'pr_f1')
    merge_eval(main_eval, pr_oracle, 'pr_oracle')


# =================================================================================================
def histogram_na_prob(na_probs, qid_list, image_dir, name):
    """
    Generates a histogram of the no answer probabilities for the questions in qid_list and
    stores the plot in the image_dir, using "name" as part of the file name and as a title
    of the plot.

    Args:

      na_probs (dictionary): dictionary that contains the no answer probabilities for each question (id)
      qid_list (list): list of question ids
      image_dir (string): the directory where the plot is to be stored
      name (string): the name that is used as part of the file name and as a title for the plot

    Returns:

      None
    """

    if not qid_list:
        return

    x = [na_probs[k] for k in qid_list]
    weights = np.ones_like(x) / float(len(x))

    plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
    plt.xlabel('Model probability of no-answer')
    plt.ylabel('Proportion of dataset')
    plt.title('Histogram of no-answer probability: %s' % name)
    plt.savefig(os.path.join(image_dir, 'na_prob_hist_%s.png' % name))
    plt.clf()


# =================================================================================================
def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    """
    This function finds the best threshold for the no answer probabilities. It works as follows:
      
        - get number of questions without ananswer
        - initialize current score, best score and best threshold
        - get a list of question ids, sorted in ascending order by their probabilities for no answer
        - for each question, update the current score with the difference between the raw score and the
          predicted answer (if the question has an answer and the model predicted one) or 0 (if the question
          has no answer or the model did not predict one)
        - if the current score is higher than the best score, update the best score and the best threshold

    Args:

      preds (dictionary):           dictionary with qid's as keys and predicted answers as values
      scores (dictionary):          contains the raw scores (either EM or F1) for all questions
      na_probs (dictionary):        no answer probability for each question (id) (generated by the model)
      qid_to_has_ans (dictionary):  for each question (id), contains True if question has answer, False else

    Returns:

      tuple: the best score and the best threshold
    """

    # get number of questions without an answer
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])  

    # Intializations
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0

    # get a list of question ids, sorted in ascending order by their probabilities
    # for no answer - i.e. the qid's with lowest probability for no answer come first
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])

    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue

        # At this point, we have a raw score for the question qid
        if qid_to_has_ans[qid]:    # if question qid has answer
            diff = scores[qid]
        else:                      # else: question has no answer...
            if preds[qid]:         # if model predicted answer
                diff = -1          # then subtract one
            else:
                diff = 0           # else, don't subtract anthing

        cur_score += diff          # update the current score with the difference

        if cur_score > best_score: # align the best score to the current score
            best_score = cur_score
            best_thresh = na_probs[qid]  # set the current probability for no answer as the new best threshold

    return 100.0 * best_score / len(scores), best_thresh


# =================================================================================================
def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    """
    Finds the best EM and F1 scores and thresholds for the no answer probabilities. The metrics
    are then stored in the main_eval dictionary.

    Args:

      main_eval (dictionary):      dictionary that holds the main evaluation metrics
      preds (dictionary):          dictionary with qid's as keys and predicted answers as values
      exact_raw (dictionary):      contains the raw exact scores for all questions
      f1_raw (dictionary):         contains the raw f1 scores for all questions
      na_probs (dictionary):       no answer probability for each question (id) (generated by the model)
      qid_to_has_ans (dictionary): for each question (id), contains True if question has answer, False else

    Returns:

      None
    """

    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


# =================================================================================================
def calc_squad_metrics(dataset, preds, na_probs=None):
    """
    This is the main function of this module. It calculates the EM and F1 scores for
      - all questions in preds
      - all questions in preds that have answers
      - all questions in preds that don't have answers

    It also generates additional statistics and plots precision-recall curves if an additional
    na_prob.json file is provided or is no answer probabilities are simulated (plots arestored in 
    "OPTS.image_dir"). This file is expected to map qid's to the model's predicted probability 
    that a question is unanswerable.

    Args:

      dataset (list):        list of articles; each entry contains data for one single article 
                            (e.g. Harvard university), including title, contexts and qas
                            (typically, read from a json file, such "dev-v2.0.json" file)

      preds (dictionary):    dictionary that holds predicitons to be evaluated, one answer
                            per question (id)

      na_probs (dictionary): per question (id), the probability that the question is unanswerable
                            (such as assessed by the model that did the predictions)

    Returns:

      out_eval (dictionary): dictionary that holds EM and F1 scores as well as totals for
                            the complete set of questions, for the subset of questions that
                            have answers and for the subset of questions that do not have
                            answers

    Example output:

    OrderedDict({'exact': 64.81091552261434, 'f1': 67.60971132981268, 'total': 11873,
    'HasAns_exact': 59.159919028340084, 'HasAns_f1': 64.76553687902599, 'HasAns_total': 5928,
    'NoAns_exact': 70.4457527333894, 'NoAns_f1': 70.4457527333894, 'NoAns_total': 5945})

    """

    # overwrite na_probs with 0.0 values, if not provided...
    if not na_probs:
        na_probs = {k: 0.0 for k in preds}

    # Get dictionary that indicates per quesion (using its id),
    # whether or not it has an answer.
    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    temp_dict = {k: qid_to_has_ans[k] for k in qid_to_has_ans if k in preds} # limit qids to qids with predictions
    qid_to_has_ans = temp_dict

    # Get the list of question (ids) that have an answer
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]

    # Get the list of questions (ids) that do *not* have an answer
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

    # Get the list of questions (ids) that have a prediction
    # Scores will only be calculated for these...
    has_pred_qids = [k for k in preds]

    # Get the EM and F1 scores for all predicted answers for the questions
    # "raw" means no threshold yet applied
    exact_raw, precision_raw, recall_raw, f1_raw = get_raw_scores(dataset, preds)

    # Update exact and f1 scores based on threshold (#TODO: describe exactly how)
    # exact_thresh and f1_thresh are also dictionaries with the question id's as 
    # keys and the scores as values
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, 1.0)
    precision_thresh = apply_no_ans_threshold(precision_raw, na_probs, qid_to_has_ans, 1.0)
    recall_thresh = apply_no_ans_threshold(recall_raw, na_probs, qid_to_has_ans, 1.0)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, 1.0)
    
    # Construct the dictionary that holds the EM, precision, recall and F1 scores for the complete predictions data set
    # example for out_eval: OrderedDict({'exact': 64.81091552261434, 'f1': 67.60971132981268, 'total': 11873})
    print("\nGetting metrics for all questions...")
    out_eval = make_eval_dict(exact_thresh, precision_thresh, recall_thresh, f1_thresh, has_pred_qids)

    # Construct the dictionary using only the questions that have answers
    # Then merge this into the out_eval dictionary
    has_ans_intersection_qids = list(set(has_pred_qids) & set(has_ans_qids))
    if has_ans_intersection_qids:
        print("Getting metrics for questions that have answers...")
        has_ans_eval = make_eval_dict(exact_thresh, precision_thresh, recall_thresh,
                                      f1_thresh, has_ans_intersection_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    else:
        print("Cannot get metrics for questions that have answers. There are none in preds...")
  
    # Construct the dictionary using only the questions that have *no* answers
    # Then also merge this into the out_eval dictionary. out_eval now contains
    # the complete info, for example:
    no_ans_intersection_qids = list(set(has_pred_qids) & set(no_ans_qids))
    if no_ans_intersection_qids:
        print("Getting metrics for questions with no answers...")
        no_ans_eval = make_eval_dict(exact_thresh, precision_thresh, recall_thresh, 
                                     f1_thresh, qid_list=list(set(has_pred_qids) & set(no_ans_qids)))
        merge_eval(out_eval, no_ans_eval, 'NoAns')
    else:
        print("Cannot get metrics for questions that don't have answers. There are none in preds...")

    if OPTS.na_prob_file or OPTS.na_prob_sim:
        print("Finding best thresholds...")
        find_all_best_thresh(out_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)

    if (OPTS.na_prob_file or OPTS.na_prob_sim) and OPTS.out_image_dir:
        print("Running further analysis and generating plots... \n")
        run_precision_recall_analysis(out_eval, exact_raw, f1_raw, na_probs,
                                      qid_to_has_ans, OPTS.out_image_dir)
        histogram_na_prob(na_probs, has_ans_qids, OPTS.out_image_dir, 'Question has answer')
        histogram_na_prob(na_probs, no_ans_qids, OPTS.out_image_dir, 'Question has no answer')

    return out_eval


# =================================================================================================
def get_qid_from_question(question, dataset):
    """
    Gets the qid of the question provided in the arg "question"

    Args:

      question(string): the string that contains the question

    Returns:

      string: the qid of the question, if it is in the dataset; an empty string else
    """

    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                if qa['question'] == question:
                    return qa['id']
    return ''


# =================================================================================================
def get_question_from_qid(qid, dataset):
    """
    Gets the questing string for the question given by the id in the arg "qid"

    Args:

      qid (string): the id of the question

    Returns:

      string: the text of the corresponding question, if it is in the dataset; an empty string else
    """

    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                if qa['id'] == qid:
                    return qa['question']
    return ''


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


# =================================================================================================
def load_preds(preds_file):
    """
    Loads the predictions file from the file specified in "preds_file" and returns a dictionary. This
    function expects a .csv file that contains at least two columns, one of them named "question" that
    contains the questions in string format (i.e. not the qid's) and the other one name "answer" that
    contains the predicted answer for each corresponding question.

    Args:

      preds_file (string): the path to the predictions file

    Returns:

      preds (dictionary): dictionary that contains all predictions, with the keys being the qid's
                          and the values the predicted answers
    """

    preds = {}

    # Load the file that contains the predictions (expected to be in csv format)
    try:
        with open(preds_file, mode="r") as file:
            reader = DictReader(file)
            for row in reader:
                qid = get_qid_from_question(row["question"], dataset)
                preds[qid] = row["answer"]
    except FileNotFoundError:
        print(f"Error: the predictions file '{preds_file}' could not be found...")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: the predictions file '{preds_file}' could not be read, since it is not a valid CSV file...")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        exit(1)
    
    if len(preds) == 0:
        print(f"The predictions could not be read. Exiting...")
        exit(1)

    return preds


# =================================================================================================
def simulate_na_probs(preds):
    """
    Simulates no answer probabilities for the questions in the preds dictionary. The probabilities
    are generated by sampling from a normal distribution with mean 0.15 for questions that have an
    answer and mean 0.85 for questions that do not have an answer. The standard deviation is 0.1. 
    The probabilities are of course clipped to the range [0.0, 1.0].

    Args:

      preds (dictionary): dictionary that contains the predictions to be evaluated

    Returns:

      na_probs (dictionary): dictionary that contains the simulated no answer probabilities for each
                              question (id) in the preds dictionary
    """

    na_probs = {}

    ans_mean = 0.15
    no_ans_mean = 0.85
    std_dev = 0.1

    for qid, ans in preds.items():
        if ans:
            sample_prob = np.random.normal(ans_mean, std_dev)
        else:
            sample_prob = np.random.normal(no_ans_mean, std_dev)

        if sample_prob > 1.0:
            sample_prob = 1.0
        elif sample_prob < 0.0:
            sample_prob = 0.0

        na_probs[qid] = sample_prob

    return na_probs


# =================================================================================================
if __name__ == '__main__':

    # Read command line parameters
    OPTS = parse_args()

    if OPTS.out_image_dir:
        matplotlib.use('Agg')

    # Overwrite cl args for the data file and for the predictions file for testing purposes
    data_file = "data/qa_dl_cache/dev-v2.0.json"
    # preds_file = "docs/evaluations/baseline-v0/baseline-evaluation-openai-results-v0.csv"
    preds_file = "data/qa_dl_cache/sample_predictions.csv"

    # Load the dataset file and the predicitons file
    dataset = load_dataset(data_file)
    preds = load_preds(preds_file)

    # Construct dictionary for no answer probabilities generated by the model that did the predictions.
    # There should be one entry per question (id) for which there is a prediction in the preds dictionary.
    # If the "na_prob_file" parameter is specified, read the probs from the file
    # TODO --> check required format of the na_prob_file
    if OPTS.na_prob_file:
        with open(OPTS.na_prob_file) as f:  # TODO: catch exception if OPTS.na_prob_file deos not exist
            na_probs = json.load(f)
    else:
        # if the file is not specified, we don't know the probs, we either simulate non answer probabilities
        # or we set probabilities to 0.0 (setting them to 0.0 ensures that apply_no_ans_threshold does
        # not change the scores)
        if OPTS.na_prob_sim:
            na_probs = simulate_na_probs(preds)
        else:
            na_probs = {k: 0.0 for k in preds}

    # Call eval_squad_preds to compute the metrics. This is the main function that calculates all
    # the metrics and generates statistics and plots.
    out_eval = calc_squad_metrics(dataset, preds, na_probs)

    # Write the results to OPTS.out_file, if provided in the cl arguments, else dump to screen
    if OPTS.out_file:
        with open(OPTS.out_file, 'w') as f:
            json.dump(out_eval, f)
    else:
        print("\nResults of the evaluation:\n")
        print(json.dumps(out_eval, indent=2), "\n")
