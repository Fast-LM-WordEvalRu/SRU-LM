import linecache

import nltk
import torch
from torch.utils.data import Dataset

from ..core.utils import raw_count


class FastDataset(Dataset):
    def __init__(self, filename, word_dict, n_samples=None, add_backward_target=False):
        self._filename = str(filename)
        self.word_dict = word_dict
        self.add_backward_target = add_backward_target
        if n_samples is None:
            self._total_data = raw_count(filename)
        else:
            self._total_data = n_samples

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        splitted_line = nltk.tokenize.word_tokenize(line)

        unk = self.word_dict['<UNK>']

        word_indices = [self.word_dict.get(w, unk) for w in splitted_line]
        forward_target = word_indices + [self.word_dict['<EOS>'], self.word_dict['<PAD>']]

        item_dict = {
            'forward_text': splitted_line,
            'forward_target': torch.LongTensor(forward_target)
        }

        if self.add_backward_target:
            backward_target = word_indices[::-1] + [self.word_dict['<BOS>'], self.word_dict['<PAD>']]
            item_dict['backward_text'] = splitted_line[::-1]
            item_dict['backward_target'] = torch.LongTensor(backward_target)

        return item_dict

    def __len__(self):
        return self._total_data
