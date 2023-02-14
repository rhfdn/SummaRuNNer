import pandas as pd
import json
import os

# Load json array undelimited by []
def load_json_as_pandas_df(fname):
    text = []
    summaries = []
    labels = []
    data = open(fname,encoding='latin-1').readlines()
    for i in range(len(data)):
        obj = json.loads(data[i])
        text.append(obj["doc"])
        summaries.append(obj["summaries"])
        tmp_labels = obj["labels"].split('\n')
        labels.append([int(tmp_labels[i]) for i in range(len(tmp_labels))])
    return pd.DataFrame(list(zip(text, summaries, labels)), columns=["text", "summaries", "labels"])

# Load all json file
ref_train = load_json_as_pandas_df("./data/ref/train.json")
ref_val = load_json_as_pandas_df("./data/ref/val.json")
ref_test = load_json_as_pandas_df("./data/ref/test.json")

# Create directory for raw dataset (csv)
if not os.path.exists("./data/raw"):
    os.makedirs("./data/raw")

ref_train.to_json("./data/raw/train.json")
ref_val.to_json("./data/raw/val.json")
ref_test.to_json("./data/raw/test.json")
