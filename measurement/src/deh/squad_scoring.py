"""

Adapted from the official evaluation script for SQuAD version 2.0.

NOTE
The following has not yet been adapted -->

  In addition to basic functionality, we also compute additional statistics and
  plot precision-recall curves if an additional na_prob.json file is provided.
  This file is expected to map question ID's to the model's predicted probability
  that a question is unanswerable.

"""

import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys

from csv import DictReader

# Global variable for command line parameters
OPTS = None

#=================================================================================================
def parse_args():
  """
  Reads in command line arguments for further processing.
  """
  parser = argparse.ArgumentParser('Official evaluation script for SQuAD version 2.0.')

  # The next two arguments are required, since they are positional...
  parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
  parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')

  # ... and these are optional
  parser.add_argument('--out-file', '-o', metavar='eval.json',
                      help='Write accuracy metrics to file (default is stdout).')
  parser.add_argument('--na-prob-file', '-n', metavar='na_prob.json',
                      help='Model estimates of probability of no answer.')
  parser.add_argument('--na-prob-thresh', '-t', type=float, default=1.0,
                      help='Predict "" if no-answer probability exceeds this (default = 1.0).')
  parser.add_argument('--out-image-dir', '-p', metavar='out_images', default=None,
                      help='Save precision-recall curves to directory.')
  parser.add_argument('--verbose', '-v', action='store_true')

  if len(sys.argv) == 1:
    parser.print_help()
    #sys.exit(1)     # Do not exit, even if no args have been supplied
  
  return parser.parse_args()


#=================================================================================================
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


#=================================================================================================
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


#=================================================================================================
def get_tokens(s):
  """
  Gets the tokens for string s
  """
  if not s: return []
  return normalize_answer(s).split()


#=================================================================================================
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


#=================================================================================================
def compute_f1(a_gold, a_pred):
  """
  Computes the f1 score, based on the gold answer and the predicted answer.

  Args:

    a_gold (string): string that contains the gold answer
    a_pred (string): string that contains the predicrted answer

  Returns:

    int: the f1 score (formula: (2 * precision * recall) / (precision + recall))

  """
  
  # Get the tokens for the gold answer and the predicted answer
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)

  # Get the common tokens and the number of common tokens
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())

  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  
  # Calculate the f1 score
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)

  return f1


#=================================================================================================
def get_raw_scores(dataset, preds):
  """
  Gets exact scores and F1 scores per question based on the dataset and the predictions

  Args:

    dataset (list): list of articles (each one containing paragraphs, each paragraph containing questions and answers)
    preds (dictionary): the predictions to be evaluated; dictionary contains one entry per question id with either an answer 
                        if one is predicted, or an empty string

  Returns:

    exact_scores (dictionary): for each question, the exact score (either 0 or 1)
    f1_score (dictionary): for each question, the f1 score (a value between 0 and 1)

  """

  exact_scores = {}
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
          #print('Missing prediction for %s' % qid)
          continue
        
        # Get the predicted answer
        a_pred = preds[qid]

        # Take max over all gold answers for exact scores and f1 scores alike
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  
  return exact_scores, f1_scores


#=================================================================================================
def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
  """
  
  Args:

    scores (dictionary): contains scores (either exact or f1), contains one entry per question (id)
    na_probs (dictionary): no answer probability for each question (id)
    qid_to_has_ans (dictionary): per question (id), contains True if answerable, False if not
    na_prob_threshold (float): threshold (1.0 by default)

  Returns:

   # TODO: comment the return value

  """
  
  new_scores = {}

  for qid, s in scores.items():
    pred_na = na_probs[qid] > na_prob_thresh
    if pred_na:
      new_scores[qid] = float(not qid_to_has_ans[qid])
    else:
      new_scores[qid] = s

  return new_scores


#=================================================================================================
def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  """
  Constructs the dictionary that holds the EM and F1 score for the complete predictions data set
  example output of the function: OrderedDict({'exact': 64.81091552261434, 'f1': 67.60971132981268, 'total': 11873})

  Args:

    exact_scores (dictionary): contains the exact scores, one per question (id)
    f1_scores (dictionary): contains the f1 scores, one per question (id)
    qid_list (list ?): #TODO comment the meaning of qid_list

  Returns:

    dictionary: dict that holds the final EM and F1 scores for the whole predictions dataset and the total number of 
                questions. EM and F1 scores are averages over all questions
                #TODO: explain the formulat that is used to calculate the F1 score

  """


  if not qid_list:
    total = len(exact_scores)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores.values()) / total),
        ('f1', 100.0 * sum(f1_scores.values()) / total),
        ('total', total),
    ])
  else:
    total = len(qid_list)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ('total', total),
    ])


#=================================================================================================
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


#=================================================================================================
def plot_pr_curve(precisions, recalls, out_image, title):

  plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
  plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.title(title)
  plt.savefig(out_image)
  plt.clf()


#=================================================================================================
def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans,
                               out_image=None, title=None):
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


#=================================================================================================
def run_precision_recall_analysis(main_eval, exact_raw, f1_raw, na_probs, 
                                  qid_to_has_ans, out_image_dir):
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


#=================================================================================================
def histogram_na_prob(na_probs, qid_list, image_dir, name):
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


#=================================================================================================
def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for i, qid in enumerate(qid_list):
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]
  return 100.0 * best_score / len(scores), best_thresh


#=================================================================================================
def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
  main_eval['best_exact'] = best_exact
  main_eval['best_exact_thresh'] = exact_thresh
  main_eval['best_f1'] = best_f1
  main_eval['best_f1_thresh'] = f1_thresh


#=================================================================================================
def eval_squad_preds(dataset, preds, na_probs):
  """
  This is the main function of this module. 

  Args:

    dataset (list): list of articles; each entry contains data for one single article 
                    (e.g. Harvard university), including title, context and qas
                    (typically, can be the "dev-v2.0.json" file, read into a list)
    preds (dictionary): dictionary that holds predicitons to be evaluated, one answer
                        per question (id)

  Returns:

    out_eval (dictionary): dictionary that holds EM and F1 scores as well as totals for
                           the complete set of questions, for the subset of questions that
                           have answers and for the subset of questions that do not have
                           answers
                
  Examples output:

  OrderedDict({'exact': 64.81091552261434, 'f1': 67.60971132981268, 'total': 11873,
  'HasAns_exact': 59.159919028340084, 'HasAns_f1': 64.76553687902599, 'HasAns_total': 5928,
  'NoAns_exact': 70.4457527333894, 'NoAns_f1': 70.4457527333894, 'NoAns_total': 5945})

  """

  # overwrite na_probs for the time being...
  #TODO --> refactor, once exact role na_probs has been understood!
  na_probs = {k: 0.0 for k in preds}

  # Get dictionary that indicates per quesion (using its id),
  # whether or not it has an answer.
  qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False

  # Get the list of question (ids) that have an answer
  has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]

  # Get the list of questions (ids) that do *not* have an answer
  no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

  # Get the list of questions (ids) that have a prediction
  # Scores will only be calculated for these...
  has_pred_qids = [k for k in preds]

  # Get the EM and F1 scores for all predicted answers for the questions
  exact_raw, f1_raw = get_raw_scores(dataset, preds)

  # Update exact and f1 scores based on threshold (TO DO: describe exactly how)
  # exact_thresh and f1_thresh are also dictionaries with the question id's as 
  # keys and the scores as values
  exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, 1.0) #OPTS.na_prob_thresh)
  f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, 1.0) #OPTS.na_prob_thresh)
  
  # Construct the dictionary that holds the EM and F1 score for the complete 
  # predictions data set
  # example for out_eval: OrderedDict({'exact': 64.81091552261434, 'f1': 67.60971132981268, 'total': 11873})
  out_eval = make_eval_dict(exact_thresh, f1_thresh, has_pred_qids)


  # Construct the dictionary using only the questions that have answers
  # Then merge this into the out_eval dictionary
  has_ans_intersection_qids = list(set(has_pred_qids) & set(has_ans_qids))
  if has_ans_intersection_qids:
    print("Getting ans metrics...")
#    print(f"1: {has_ans_intersection_qids}")
    has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, has_ans_intersection_qids)
                                  #qid_list=has_pred_qids) #qid_list=has_ans_qids)
    merge_eval(out_eval, has_ans_eval, 'HasAns')
  else:
    print("Cannot get ans metrics...")
 
  # Construct the dictionary using only the questions that have *no* answers
  # Then also merge this into the out_eval dictionary. out_eval now contains
  # the complete info, for example:
  # OrderedDict({'exact': 64.81091552261434, 'f1': 67.60971132981268, 'total': 11873,
  # 'HasAns_exact': 59.159919028340084, 'HasAns_f1': 64.76553687902599, 'HasAns_total': 5928,
  # 'NoAns_exact': 70.4457527333894, 'NoAns_f1': 70.4457527333894, 'NoAns_total': 5945})
  no_ans_intersection_qids = list(set(has_pred_qids) & set(no_ans_qids))
  if no_ans_intersection_qids:
    print("Getting no ans metrics...")
#    print(f"2: {no_ans_intersection_qids}")
    no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=list(set(has_pred_qids) & set(no_ans_qids))) #qid_list=no_ans_qids)
    merge_eval(out_eval, no_ans_eval, 'NoAns')
  else:
    print("Cannot get no ans metrics...")

  return out_eval


#=================================================================================================
def get_qid(question, dataset):
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


#=================================================================================================
if __name__ == '__main__':

  OPTS = parse_args()

  #TODO: currently, matplotlib cannot be used due to a dependency conflict (probably linked to Python 3.13)
  #TODO: create new environment cloned from current one, but with Python 3.12; then install matplotlib
  #TODO: investigate all functions that compute additional statistics and plot precision-recall curves
  # if OPTS.out_image_dir:
  #   import matplotlib
  #   matplotlib.use('Agg')
  #   import matplotlib.pyplot as plt 

  print(f"OPTS --> {OPTS}") 

  # Overwrite cl args for testing purposes
  data_file = "data/qa_dl_cache/dev-v2.0.json"
  preds_file = "docs/evaluations/baseline-v0/baseline-evaluation-openai-results-v0.csv"

  # Load the file that contains the dataset (expected to be in json format)
  with open(data_file) as f:
    dataset_json = json.load(f)         # dataset_json: dict with 'version' and 'data' as keys
                                        # 'data' contains the real data (see next variable)
    dataset = dataset_json['data']      # list of articles; each entry contains data for one single
                                        # article (e.g. Harvard university), including title, context 
                                        # and qas

  # Load the file that contains the predictions  
  preds = {}
  with open(preds_file, mode="r") as file:
    reader = DictReader(file)
    for row in reader:
      qid = get_qid(row["question"], dataset)
      preds[qid] = row["answer"]

  # construct dictionary for no answers
  # one entry per question      
  if OPTS.na_prob_file:
    with open(OPTS.na_prob_file) as f:
      na_probs = json.load(f)
  else:
    na_probs = {k: 0.0 for k in preds}

  # Call eval_squad_preds to compute the metrics
  out_eval = eval_squad_preds(dataset, preds, na_probs)

  print(out_eval)
