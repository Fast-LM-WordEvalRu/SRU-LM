import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.optim import Adam

from fast_elmo.load_data.dataset_ner import NERDataset
from fast_elmo.load_data.dataloader import get_dataloader


class EvaluatorNER:
    def __init__(self, path_to_ner, lm_model, cuda=True, train_epochs=10):
        device_name = 'cuda' if cuda else 'cpu'
        self.device = torch.device(device_name)

        train_dataset = NERDataset(path_to_ner / 'train.txt')
        test_dataset = NERDataset(path_to_ner / 'test.txt')

        self.train_dataloader = get_dataloader(train_dataset, {'PAD': 0})
        self.test_dataloader = get_dataloader(test_dataset, {'PAD': 0})

        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=11)  # у нас 11 уникальных меток для слов
        ).to(self.device)

        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = Adam(self.head.parameters())
        self.lm_model = lm_model
        self.train_epochs = train_epochs

    def train(self):
        history = []
        for epoch in range(self.train_epochs):
            losses = []
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()

                ids, mask, targets = batch
                ids = ids.to(self.device)
                mask = mask.to(self.device)
                targets = targets.flatten().to(self.device)

                with torch.no_grad():
                    model_out = self.lm_model(ids, mask)

                out = self.head(model_out).flatten(0, 1)
                loss = self.loss_func(out, targets)

                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
            history.append(losses)
        return history

    def evaluate(self):
        preds = []
        ground_truth = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                ids, mask, targets = batch

                ids = ids.to(self.device)
                mask = mask.to(self.device)
                targets = targets.flatten()

                model_out = self.lm_model(ids, mask)
                out = self.head(model_out).flatten(0, 1)

                ground_truth.append(targets)
                preds.append(out)

        preds = np.hstack([p.argmax(axis=1).cpu().numpy() for p in preds])
        ground_truth = np.hstack([g.numpy() for g in ground_truth])

        f1 = f1_score(y_true=ground_truth, y_pred=preds,
                      average='weighted', labels=np.arange(1, 11), zero_division=0)

        return f1
