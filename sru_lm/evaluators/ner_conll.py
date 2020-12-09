import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sru_lm.core import batch_to_ids

from sru_lm.load_data.dataset_ner import NERDataset
from ..models.ner import CRFModel

import os
if os.environ.get('RUN_FROM_JUPYTER', 'False') == 'True':
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange




def transform_batch(in_dict):
    out_dict = {}
    out_dict['target'] = pad_sequence([d['target'] for d in in_dict], padding_value=0, batch_first=True)
    text_ids, mask = batch_to_ids([d['text'] for d in in_dict])
    out_dict['text_ids'] = text_ids
    out_dict['mask'] = mask
    return out_dict


class EvaluatorNER:
    def __init__(self, path_to_ner, ner_labels, lm_model, cuda=True, train_epochs=5, batch_size=64):
        device_name = 'cuda' if cuda else 'cpu'
        self.device = torch.device(device_name)

        train_dataset = NERDataset(path_to_ner / 'train.txt', ner_labels)
        test_dataset = NERDataset(path_to_ner / 'test.txt', ner_labels)

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=False, collate_fn=transform_batch)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False, collate_fn=transform_batch)

        self.head = CRFModel(ner_labels, emb_size=1024).to(self.device)
        self.optimizer = Adam(self.head.parameters(), lr=1e-3)
        self.lm_model = lm_model
        self.train_epochs = train_epochs
        self.ner_labels = ner_labels

    def train(self):
        self.head.train()
        history = []
        for epoch in trange(self.train_epochs, desc='NER Train Epochs', leave=False):
            losses = []
            for batch in tqdm(self.train_dataloader, desc='NER train batch', leave=False):
                for key in batch.keys():
                    batch[key] = batch[key].cuda()

                self.optimizer.zero_grad()

                ids = batch['text_ids']
                mask = batch['mask']
                targets = batch['target']

                with torch.no_grad():
                    embeddings = self.lm_model(ids, mask, return_embeddings=True)

                loss = self.head(embeddings, mask, targets)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
            history.append(losses)
        return history

    def evaluate(self):
        self.head.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc='NER eval batch', leave=False):
                for key in batch.keys():
                    batch[key] = batch[key].cuda()

                ids = batch['text_ids']
                mask = batch['mask']
                targets = batch['target'].flatten()

                embeddings = self.lm_model(ids, mask, return_embeddings=True)
                out = self.head(embeddings, mask, targets)

                all_preds = []
                all_ground_truth = []
                for i, y_pred in enumerate(out):
                    all_preds.append(y_pred)
                    all_ground_truth.append(batch['target'][i, :len(y_pred)].cpu().numpy())

        y_true = np.concatenate(all_ground_truth)
        y_pred = np.concatenate(all_preds)

        f1 = f1_score(y_true=y_true, y_pred=y_pred,
                      average='weighted', labels=np.arange(1, len(self.ner_labels)), zero_division=0)

        return f1
