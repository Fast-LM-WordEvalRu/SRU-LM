#   Author: Artem Skiba
#   Created: 20/01/2020

from pathlib import Path

from torch.utils.data import DataLoader
from .vectorizer import Vectorizer
from .dataset import FastDataset


class BatchGenerator(object):
    def __init__(self, path_to_data: Path, batch_size: int = 16,
                 shuffle: bool = True, drop_last: bool = True):
        """
        :param path_to_data: path to folder with train.txt, dev.txt, test.txt files
        """
        path_to_train = path_to_data / 'train.txt'  # TODO: переписать это все без копипасты кода
        path_to_dev = path_to_data / 'dev.txt'
        path_to_test = path_to_data / 'test.txt'

        train_vectorizer = Vectorizer.from_text_file(path_to_train)

        train_dataset = FastDataset(path_to_train, train_vectorizer)
        dev_dataset = FastDataset(path_to_dev, train_vectorizer)
        test_dataset = FastDataset(path_to_test, train_vectorizer)

        self.__train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
        self.__dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=batch_size,
                                           shuffle=shuffle, drop_last=drop_last)
        self.__test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            shuffle=shuffle, drop_last=drop_last)

        self.train_vectorizer = train_vectorizer

        self.train_generator = None
        self.dev_generator = None
        self.test_generator = None
        self.initialize_generators()

    def generate_train_batch(self):
        return next(self.train_generator)

    def generate_dev_batch(self):
        return next(self.dev_generator)

    def generate_test_batch(self):
        return next(self.test_generator)

    def initialize_generators(self):
        self.train_generator = self.__initialize_train_generator()
        self.dev_generator = self.__initialize_dev_generator()
        self.test_generator = self.__initialize_test_generator()

    def __initialize_train_generator(self):
        for data_dict in self.__train_dataloader:
            yield data_dict

    def __initialize_dev_generator(self):
        for data_dict in self.__dev_dataloader:
            yield data_dict

    def __initialize_test_generator(self):
        for data_dict in self.__test_dataloader:
            yield data_dict
