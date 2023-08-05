import pandas as pd

# Compute the average proportion of sentences per document
def compute_the_average_proportion_of_sentences_per_document(df):
    sum_proportion = 0
    for idx in df.index:
        sum_proportion += sum(df["own_labels"][idx]) / len(df["own_labels"][idx])
    return sum_proportion / len(df.index)

print(compute_the_average_proportion_of_sentences_per_document(pd.read_json("./data/cnn_dailymail/train.json")))
