import pandas as pd
import json
import os
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import LineTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load raw dataset
raw_train = pd.read_json("./data/cnn_dailymail/pre/train.json")
raw_val = pd.read_json("./data/cnn_dailymail/pre/val.json")
raw_test = pd.read_json("./data/cnn_dailymail/pre/test.json")

# Compute own label for entry (text)
def compute_own_labels_text(text, summary, is_sep_n = False):
    labels = []
    s = []
    # tokenize sentence
    if is_sep_n:
        nltk_line_tokenizer = LineTokenizer()
        s = nltk_line_tokenizer.tokenize(text)
    else:
        s = sent_tokenize(text)
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
def compute_own_labels_df(df, title=""):
    labels = []

    articles = df["article"].tolist()
    highlights = df["highlights"].tolist()

    with tqdm(list(range(df.shape[0])), unit="article", total=df.shape[0]) as t:
        for i in t:
            if len(title) > 0:
                t.set_description(title)
            labels.append(compute_own_labels_text(articles[i], highlights[i]))
    return pd.Series(labels)

raw_test["labels"] = compute_own_labels_df(raw_test, "test set")
raw_val["labels"] = compute_own_labels_df(raw_val, "val test")
raw_train["labels"] = compute_own_labels_df(raw_train, "train test")

raw_train.to_json("./data/cnn_dailymail/train.json")
raw_val.to_json("./data/cnn_dailymail/val.json")
raw_test.to_json("./data/cnn_dailymail/test.json")
