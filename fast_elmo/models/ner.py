import torch
from torchcrf import CRF


class CRFModel(torch.nn.Module):
    def __init__(self, ner_labels, emb_size):
        super(CRFModel, self).__init__()
        self.lstm = torch.nn.LSTM(emb_size, 200, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self.line = torch.nn.Linear(400, len(ner_labels))
        self.crf = CRF(num_tags=len(ner_labels), batch_first=True)

    def forward(self, embeddings, mask, y=None):
        transformed = self.dropout1(embeddings)
        transformed, _ = self.lstm(transformed)
        transformed = self.dropout2(transformed)
        transformed = self.relu(transformed)
        transformed = self.line(transformed)

        if self.training:
            out = self.crf(transformed, y, mask) * (-1)
        else:
            out = self.crf.decode(transformed, mask)
        return out
