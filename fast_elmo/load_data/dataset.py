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

        forward_target = [self.word_dict[w] for w in splitted_line]
        forward_target += [self.word_dict['EOS'], self.word_dict['PAD']]

        if self.add_backward_target:
            pass  # TODO add backward target

        return {
            'raw_text': splitted_line,
            'forward_target': torch.LongTensor(forward_target)
        }

    def __len__(self):
        return self._total_data
