from .preprocess_text import preprocess_text

# Preprocess a dataframe: Return an array of {doc preprocessed, labels}
def preprocess_df(df, glovemgr, doc_column_name="text", labels_column_name="own_labels", is_sep_n = True, remove_stop_word = True, stemming=True, trunc_sent=-1, padding_sent=-1, trunc_doc=-1):
    result = []
    for idx in df.index:
        result.append({"idx" : idx, "doc" : preprocess_text(df[doc_column_name][idx], glovemgr=glovemgr, is_sep_n=is_sep_n, remove_stop_word=remove_stop_word, stemming=stemming, trunc_sent=trunc_sent, padding_sent=padding_sent), "labels" : df[labels_column_name][idx]})
        if trunc_doc >= 0:
            result[-1] = {"idx" : idx, "doc" : result[-1]["doc"][:min(len(result[-1]["doc"]), trunc_doc)], "labels" : result[-1]["labels"][:min(len(result[-1]["labels"]), trunc_doc)]}
    return result
