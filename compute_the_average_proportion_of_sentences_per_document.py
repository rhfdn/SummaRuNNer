import pandas as pd

# Compute the average proportion of sentences per document
def compute_the_average_proportion_of_sentences_per_document(df):
    sum_proportion = 0
    for idx in df.index:
        sum_proportion += sum(df["labels"][idx]) / len(df["labels"][idx])
    return sum_proportion / len(df.index)

print(compute_the_average_proportion_of_sentences_per_document(pd.read_json("./data/nyt_corpus_LDC2008T19_50.json")))
