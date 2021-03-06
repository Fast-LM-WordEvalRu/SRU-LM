import nltk
import torch
from torch.utils.data import Dataset

from ..core.utils import raw_count


class StraightReadDataset(Dataset):
    def __init__(self, filename, word_dict, n_samples=None, max_sent_len=None):
        self._filename = str(filename)
        self.fileobj = None

        self.word_dict = word_dict
        self.max_sent_len = max_sent_len
        self.prev_idx = None
        if n_samples is None:
            self.corpus_len = raw_count(filename)
        else:
            self.corpus_len = n_samples

    def __getitem__(self, idx):
        assert idx >= 0
        if self.prev_idx is None or idx < self.prev_idx:
            if self.fileobj:
                self.fileobj.close()

            self.fileobj = open(self._filename)
            lines_to_skip = idx
        else:
            lines_to_skip = idx - self.prev_idx - 1

        while lines_to_skip > 0:
            if self.fileobj.readline().strip():
                lines_to_skip -= 1

        line = self.fileobj.readline()
        self.prev_idx = idx
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
