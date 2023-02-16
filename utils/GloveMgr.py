# URLs:
# * load glove embedding: https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a

import numpy as np

# Manage embedding and conversion between word and index
class GloveMgr():
    def __init__(self, fname, vocab_size=150000):
        vocab,embeddings = [],[]

        with open(fname,'rt') as fi:
            full_content = fi.read().strip().split('\n')

        for i in range(min(vocab_size, len(full_content))):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab.append(i_word)
            embeddings.append(i_embeddings)

        vocab_npa = np.array(vocab)
        embs_npa = np.array(embeddings)
        # insert '<pad>' and '<unk>' tokens at start of vocab_npa.
        vocab_npa = np.insert(vocab_npa, 0, '<pad>')
        self.vocab_npa = np.insert(vocab_npa, 1, '<unk>')

        pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
        unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

        # insert embeddings for pad and unk tokens at top of embs_npa.
        self.embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))

        # quick find id of word
        self.word2id = {self.vocab_npa[i] : i for i in range(self.vocab_npa.shape[0])}

    # Convert word to index
    def w2i(self, w):
        if w in self.word2id.keys():
            return self.word2id[w]
        return 1

    # Convert index to word
    def i2w(self, i):
        if i < self.vocab_npa:
            return self.vocab_npa[i]
        return self.vocab_npa[1]

    # Get embedding
    def getEmbedding(self):
        return self.embs_npa
