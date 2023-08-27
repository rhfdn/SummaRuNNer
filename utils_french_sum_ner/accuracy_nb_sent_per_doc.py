import math
import numpy as np

from .threshold_probs import threshold_probs_by_nb

# Compute the accuracy
def accuracy_nb_sent_per_doc_fn(probs, targets, doc_lens, average_number_of_sentences_per_document=6.061850780738518):
    result = threshold_probs_by_nb(probs=probs, doc_lens=doc_lens, average_number_of_sentences_per_document=average_number_of_sentences_per_document)
    return sum([result[i] == targets[i] for i in range(len(targets))]) / len(targets)
