import linecache

import nltk
import torch
from torch.utils.data import Dataset

from ..core.utils import raw_count


class FastDataset(Dataset):
    def __init__(self, filename, word_dict, n_samples=None, max_sent_len=None):
        self._filename = str(filename)
        self.word_dict = word_dict
        self.max_sent_len = max_sent_len
        if n_samples is None:
            self.corpus_len = raw_count(filename)
        else:
            self.corpus_len = n_samples

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        splitted_line = nltk.tokenize.word_tokenize(line)

        if self.max_sent_len and len(splitted_line) > self.max_sent_len:
            splitted_line = splitted_line[:self.max_sent_len]

        unk = self.word_dict['<UNK>']
        num = self.word_dict['<NUM>']

        word_indices = [self.word_dict['<BOS>']] + [self.word_dict.get(w, unk) if not w.isdigit() else num for w in splitted_line] + [self.word_dict['<EOS>'], self.word_dict['<PAD>']]
        return {
            'text': splitted_line, 
            'word_indices': torch.LongTensor(word_indices)
            }

    def __len__(self):
        return self.corpus_len
