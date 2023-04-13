import math
import numpy as np

from .threshold_probs import threshold_probs_by_prop

# Compute the accuracy
def accuracy_prop_sent_per_doc_fn(probs, targets, doc_lens, average_proportion_of_sentences_per_document):
    result = threshold_probs_by_prop(probs, doc_lens, average_proportion_of_sentences_per_document)
    return sum([result[i] == targets[i] for i in range(len(targets))]) / len(targets)
