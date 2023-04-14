from nltk.tokenize import LineTokenizer, sent_tokenize

# Split one document
def split_doc(doc, is_sep_n = True):
    result = doc
    
    # tokenize sentence
    if is_sep_n:
        nltk_line_tokenizer = LineTokenizer()
        result = nltk_line_tokenizer.tokenize(result)
    else:
        result = sent_tokenize(result)

    # lower
    result = [line.lower() for line in result]

    return result

# Split all document in the array
def split_all_docs(docs, is_sep_n = True):
    result = []
    for doc in docs:
        result.append(split_doc(doc=doc, is_sep_n=is_sep_n))
    return result
