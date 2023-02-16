from .preprocess_text import preprocess_text

# Preprocess a dataframe: Return an array of {doc preprocessed, labels}
def preprocess_df(df, glovemgr, doc_column_name="text", labels_column_name="own_labels", is_sep_n = True, remove_stop_word = True, stemming=True, trunc=-1, padding=-1):
    result = []
    for idx in df.index:
        result.append({"doc" : preprocess_text(df[doc_column_name][idx], glovemgr=glovemgr, is_sep_n=is_sep_n, remove_stop_word=remove_stop_word, stemming=stemming, trunc=trunc, padding=padding), "labels" : df[labels_column_name][idx]})
    return result
