from collections import defaultdict
import re
import ast
import base64
import io
import random
import numpy as np
import os
import json
import logging
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

lmms_logger = logging.getLogger("lmms-eval")

OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase. If no answer is available, type '<no answer>'."


def construct_prompt(doc):
    question = doc["question"]
    # question = f"{question}\n{OPEN_ENDED_PROMPT}"
    question = f"{OPEN_ENDED_PROMPT}\n{question}"
    return question


def screenqa_doc_to_text(doc):
    question = construct_prompt(doc)
    return question


def screenqa_doc_to_visual(doc):
    img = doc["image"]
    return [img]


def screenqa_process_results(doc, results):
    pred = results[0]
    parsed_pred = pred
    id = doc["screen_id"]
    ans = {"id": id, "answer": doc["ground_truth"], "parsed_pred": parsed_pred}
    return {
        "screenqa_f1": ans,
        "submission": {
            id: pred,
        },
    }


def screenqa_aggregate_results(results):
    # import code 
    # code.interact(local=locals())
    printable_results = {}
    printable_results["Overall"] = {
        "num": len(results),
        "f1": round(evaluate_screenqa_short(results)[1]['f1'], 3),
    }
    print(printable_results)
    return printable_results["Overall"]["f1"]


def evaluate_screenqa_short(samples):

    def _normalize_str(string):
        # lower it
        string = string.lower()

        # strip non-alphanumeric characters
        string = re.sub(r"[^a-zA-Z0-9]", "", string)

        # strip leading and trailing whitespaces
        string = string.strip()
        
        return string

    def _compute_f1(sa, sb):
        sa = _normalize_str(sa)
        sb = _normalize_str(sb)

        if len(sa) == 0 or len(sb) == 0:
            return 0.0

        comm = set(sa).intersection(set(sb))
        prec = len(comm) / len(sb)
        rec = len(comm) / len(sa)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        return f1

    judge_list = []
    for sample in samples:
        f1_i = max([
            _compute_f1(ans, sample["parsed_pred"])
            for ans in sample["answer"]
        ])
        judge_list.append(f1_i)

    f1 = np.mean(judge_list)
    return judge_list, {"f1": f1}
