import math
import numpy as np

# Compute the accuracy
def accuracy_nb_sent_per_doc_fn(probs, targets, doc_lens, average_number_of_sentences_per_document=6.061850780738518):
    result = []
    doc_i = 0
    doc_len = 0
    doc_line_i = 0
    while doc_i < len(doc_lens):
        doc_len = doc_lens[doc_i]
        doc_prob = np.array([probs[i] for i in range(len(probs))])
        n = math.ceil(average_number_of_sentences_per_document)
        for i in range(n):
            idx = np.argmax(doc_prob)
            doc_prob[idx] = -1
        doc_prob[doc_prob >= 0] = 0
        doc_prob[doc_prob == -1] = 1
        result = result + [doc_prob[i] for i in range(len(probs))]
        doc_i += 1
        doc_line_i += doc_len
    return sum([result[i] == targets[i] for i in range(len(targets))]) / len(targets)
