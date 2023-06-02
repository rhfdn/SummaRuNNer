import pandas as pd

# Compute the average number of sentences per document
def compute_the_average_number_of_sentences_per_document(df):
    sum_number = 0
    for idx in df.index:
        sum_number += sum(df["own_labels"][idx])
    return sum_number / len(df.index)

print(compute_the_average_number_of_sentences_per_document(pd.read_json("./data/train.json")))
