"""
Vocabulary: класс, который осуществляет преобразование слова в число.
Сейчас этот класс будет просто оберткой над словарем.
"""


class Vocabulary(object):
    """
    Class for transforming word to word index
    """
    # TODO: use DefaultDict instead of Dict
    # TODO: use '[]' operators instead of add_token and lookup_token methods
    # TODO: implement special tokens for numbers (<NUM> and maybe <YEAR>)

    def __init__(self, token_to_idx=None,
                 add_unk=True, unk_token='<UNK>',
                 add_start_end=True, start_token='<START>', end_token='<END>'):

        if token_to_idx is None:
            token_to_idx = dict()

        self.__token_to_idx = token_to_idx
        self.__idx_to_token = {idx: token for token, idx in self.__token_to_idx.items()}

        self.__add_unk = add_unk
        self.__unk_token = unk_token

        self.__add_start_end = add_start_end
        self.__start_token = start_token
        self.__end_token = end_token

        self.__unk_index = -1
        if self.__add_unk:
            self.__unk_index = self.add_token(unk_token)

        self.__start_index = -1
        self.__end_index = -1
        if self.__add_start_end:
            self.__start_index = self.add_token(start_token)
            self.__end_index = self.add_token(end_token)

    def add_token(self, token):
        """
        Update mapping dicts based on the token
        """
        if token in self.__token_to_idx:
            index = self.__token_to_idx[token]
        else:
            index = len(self.__token_to_idx)
            self.__token_to_idx[token] = index
            self.__idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """
        Retrieve the index associated with the token
        or the UNK index if token isn't present
        """
        if self.__unk_index >= 0:
            return self.__token_to_idx.get(token, self.__unk_index)
        else:
            return self.__token_to_idx[token]

    def lookup_index(self, index):
        """
        Return the token associated with the index
        """
        if index not in self.__idx_to_token:
            raise KeyError(f'the index {index} is not in the Vocabulary')
        else:
            return self.__idx_to_token[index]

    @property
    def start_index(self) -> int:
        """
        :return: index of a <START> token
        """
        return self.__start_index

    @property
    def end_index(self) -> int:
        """
        :return: :return: index of a <END> token
        """
        return self.__end_index
