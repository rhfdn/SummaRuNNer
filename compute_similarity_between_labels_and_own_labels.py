import pandas as pd

def similarity(arr1, arr2):
    a = []
    for i in range(len(arr1)):
        a.append(1 if arr1[i] == arr2[i] else 0)
    return sum(a) / len(a)

df = pd.read_json("./data/test.json")

s = 0
for i in df.index:
    s += similarity(df["labels"][i], df["own_labels"][i])


print("Similarity =", s/len(df.index))

