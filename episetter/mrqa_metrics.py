"""SHIFT scoring functions; see THIRD_PARTY.md and licenses/SHIFT-LICENSE."""

import re
import string
from collections import Counter
from typing import Optional

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"\u2018", u"\u2019", u"\u00b4", u"\u0060"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    def replace_underscore(text):
        return text.replace('_', ' ')
    return white_space_fix(remove_articles(handle_punc(replace_underscore(s.lower())))).strip()

def extract_answer(solution_str: str) -> Optional[str]:
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def acc_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths) -> float:
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]
    return max(metric_fn(prediction, gt) for gt in ground_truths)
