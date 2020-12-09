import json
from pathlib import Path

import numpy as np
from tqdm import trange
import torch
from torch.nn import AdaptiveLogSoftmaxWithLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from fast_elmo.load_data.dataset import FastDataset
from fast_elmo.load_data.dataloader import get_dataloader
from fast_elmo.models.bidirectional_lm import BidirectionalLM
from fast_elmo.models.trainer import train_language_model, evaluate_language_model
from fast_elmo.evaluators.ner_conll import EvaluatorNER


class WriterMock:
    def add_scalar(self, name, value, x):
        # print(f'{name} value: {str(value)}   Epoch: {str(x)}')
        pass

    def add_scalars(self, name, value, x):
        self.add_scalar(name, value, x)

    def close(self):
        pass


if __name__ == '__main__':

    path_to_data = Path('../data/Lenta_ru -- cleaned/')

    with (path_to_data / 'lentaru_word_dict.json').open() as f:
        word_dict = json.load(f)

    train_dataset = FastDataset(path_to_data / 'train.txt', word_dict, n_samples=1_000_000)
    dev_dataset = FastDataset(path_to_data / 'dev.txt', word_dict, n_samples=1_000)

    train_dataloader = get_dataloader(train_dataset, word_dict, bidirectional=True)
    dev_dataloader = get_dataloader(dev_dataset, word_dict, bidirectional=True)

    model = BidirectionalLM(sru=False).cuda()

    path_to_ner = Path('/home/artem/DataScience/Wikiner/')
    ner_labels = ['O', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
    evaluator = EvaluatorNER(path_to_ner, ner_labels, model, batch_size=16, train_epochs=10)

    cutoffs = [100, 1000, 5000]

    loss_forward = AdaptiveLogSoftmaxWithLoss(in_features=512, n_classes=len(word_dict.values()), cutoffs=cutoffs)
    loss_backward = AdaptiveLogSoftmaxWithLoss(in_features=512, n_classes=len(word_dict.values()), cutoffs=cutoffs)
    loss_forward.cuda()
    loss_backward.cuda()

    optimizer = Adam(model.parameters(), lr=5e-4)

    writer = SummaryWriter()
    # writer = WriterMock()

    n_epochs = 9
    evaluate_after = 3

    path_to_save_model = 'bidirectional_lm'

    for epoch in trange(n_epochs, desc='Epochs'):
        # Language model
        train_losses = train_language_model(model, loss_forward, loss_backward, optimizer, train_dataloader)
        eval_losses, perplexies = evaluate_language_model(model, loss_forward, dev_dataloader)

        writer.add_scalars('Language Model/loss',
                           {'train': np.mean(train_losses), 'eval': np.mean(eval_losses)},
                           epoch)

        writer.add_scalar('Language Model/eval perplexity', np.mean(perplexies), epoch)

        # Evaluating on NER task

        if (epoch+1) % evaluate_after == 0:
            train_losses = evaluator.train()
            f1 = evaluator.evaluate()

            writer.add_scalars('NER/loss',
                               {'train': np.mean(train_losses), 'eval': np.mean(eval_losses)},
                               epoch)

            writer.add_scalar('NER/eval F1 score', np.mean(f1), epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': (loss_forward, loss_backward),
        }, path_to_save_model)

    writer.close()
