import pandas as pd
import json
import os
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load raw dataset
raw_train = pd.read_json("./data/raw/train.json")
raw_val = pd.read_json("./data/raw/val.json")
raw_test = pd.read_json("./data/raw/test.json")

# Compute own label for entry (text)
def compute_own_labels_text(text, summary, sep='\n'):
    labels = []
    s = text.split(sep)
    if (len(s) > 0):
        a = ""
        score = scorer.score(a, summary)
        for i in range(len(s)):
            current_score = scorer.score(a + s[i] + ".", summary)
            if  current_score["rouge1"].fmeasure > score["rouge1"].fmeasure or \
                current_score["rouge2"].fmeasure > score["rouge2"].fmeasure or \
                current_score["rougeL"].fmeasure > score["rougeL"].fmeasure:
                score = current_score
                a = a + s[i] + "."
                labels.append(1)
            else:
                labels.append(0)
    return labels

# Compute own label for dataframe
def compute_own_labels_df(df):
    labels = []
    for i in range(df.shape[0]):
        labels.append(compute_own_labels_text(df["text"][i], df["summaries"][i]))
    return pd.Series(labels)

raw_train["own_label"] = compute_own_labels_df(raw_train)
raw_val["own_label"] = compute_own_labels_df(raw_val)
raw_test["own_label"] = compute_own_labels_df(raw_test)

raw_train.to_json("./data/train.json")
raw_val.to_json("./data/val.json")
raw_test.to_json("./data/test.json")
