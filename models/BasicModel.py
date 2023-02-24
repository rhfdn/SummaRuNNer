import torch
import torch.nn as nn

class BasicModel(nn.Module):
    def __init__(self, device = None):
        super(BasicModel, self).__init__()
        self.device = device

    def pad_doc(self, x, doc_lens):
        pad_dim = x.size(1)
        max_doc_len = max(doc_lens)
        result = []
        start = 0
        for doc_len in doc_lens:
            stop = start + doc_len
            doc = x[start:stop]
            start = stop
            if doc_len == max_doc_len:
                result.append(doc.unsqueeze(0))
            else:
                pad = torch.zeros(max_doc_len-doc_len, pad_dim)
                if self.device is not None:
                    pad = pad.to(self.device)
                result.append(torch.cat([doc,pad]).unsqueeze(0))
        result = torch.cat(result,dim=0)
        return result

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
