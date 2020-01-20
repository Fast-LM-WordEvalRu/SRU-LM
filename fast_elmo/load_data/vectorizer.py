#   Author: Artem Skiba
#   Created: 20/01/2020

from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from fast_elmo.core.utils import split
from .vocabulary import Vocabulary


class Vectorizer(object):
    """
    Vectorizer: класс, осуществляющий преобразование текстовой строки в последовательность чисел.
    Также он гарантирует, что эти последовательности чисел будут иметь одну и ту же длину
    (сейчас это реализовано очень костыльно, в будущем нужно будет сделать гибкое изменение размера).
    Осуществляет часть работы "подсчет частот встречаемости слов, выкидывание редких слов и создание Vocabulary"
    """

    def __init__(self, train_vocab: Vocabulary, num_samples: int):
        self.train_vocab = train_vocab
        self.num_samples = num_samples

    def vectorize(self, text: list):
        """
        Create a numpy vector with indices of tokens in texts
        """
        vectorized_texts = []

        max_text_len = 100  # TODO: don't use particular number

        vectorized_text = [self.train_vocab.start_index]  # TODO: implement case without start and end tokens

        for word in split(text):
            word_idx = self.train_vocab.lookup_token(word)
            vectorized_text.append(word_idx)

        vectorized_text.append(self.train_vocab.end_index)
        vectorized_text.extend([0] * (max_text_len - len(vectorized_text)))

        return vectorized_text

    @classmethod
    def from_text_file(cls, path_to_file: Path, cutoff=1):
        """
        Instantiate the vectorizer from text file
        Words with frequency equal of less than cutoff won't be added in words dictionary
        """
        word_count = defaultdict(int)
        num_samples = 0

        with path_to_file.open() as f:
            for line in tqdm(f.readlines()):
                num_samples += 1
                for word in split(line):
                    word_count[word] += 1

        word_dict = dict()

        current_word_idx = 1

        for word in word_count.keys():
            if word_count[word] > cutoff:
                word_dict[word] = current_word_idx
                current_word_idx += 1

        return cls(train_vocab=Vocabulary(word_dict), num_samples=num_samples)
