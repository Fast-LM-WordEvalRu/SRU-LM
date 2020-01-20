"""
Датасет, который
"""

import linecache
import numpy as np
from torch.utils.data import Dataset

from src.core.utils import split, raw_count
from .vectorizer import Vectorizer


class FastDataset(Dataset):
    def __init__(self, filename, vectorizer):
        self._filename = str(filename)
        self._vectorizer = vectorizer
        self._total_data = raw_count(filename)

    def __getitem__(self, idx):
        line = ' '.join(split(linecache.getline(self._filename, idx + 1)))
        line_vectorized = self._vectorizer.vectorize(line)

        return {'raw_text': line,
                'forward_target': np.array(line_vectorized[1:] + [0]),
                'backward_target': np.array([0] + line_vectorized[:-1])
                }

    def __len__(self):
        return self._total_data
