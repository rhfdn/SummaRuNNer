from .preprocess_text import preprocess_text

# Preprocess a dataframe: Return an array of {doc preprocessed, labels}
def preprocess_df(df, glovemgr, doc_column_name="text", labels_column_name="own_labels", is_sep_n = True, remove_stop_word = True, stemming=True, trunc_sent=-1, padding_sent=-1, trunc_doc=-1):
    result = []
    for i in range(df.shape[0]):
        idx = df.index[i]
        result.append({"idx" : idx, "docs" : preprocess_text(df.iloc[i][doc_column_name], glovemgr=glovemgr, is_sep_n=is_sep_n, remove_stop_word=remove_stop_word, stemming=stemming, trunc_sent=trunc_sent, padding_sent=padding_sent), "labels" : df.iloc[i][labels_column_name]})
        if trunc_doc >= 0:
            result[-1] = {"idx" : idx, "docs" : result[-1]["docs"][:min(len(result[-1]["docs"]), trunc_doc)], "labels" : result[-1]["labels"][:min(len(result[-1]["labels"]), trunc_doc)]}
    return result
