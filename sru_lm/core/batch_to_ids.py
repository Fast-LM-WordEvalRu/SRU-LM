import re

import torch


def batch_to_ids(batch):
    if not hasattr(batch_to_ids, 'text_transformer'):
        batch_to_ids.text_transformer = TextTransformer()
    return batch_to_ids.text_transformer.batch_to_ids(batch)


class TextTransformer:
    def __init__(self):
        self.max_characters_per_token = 25  # в русском языке почти нет слов длиннее 20 букв
        self.model_char_dict = TextTransformer.build_char_dict()
        self.max_char_idx = max(self.model_char_dict.values())

    def batch_to_ids(self, batch):
        sentence_ids = [self.sentence_to_ids(sentence) for sentence in batch]
        max_sentence_len = max(len(s) for s in sentence_ids)+1

        mask = []
        ids = []

        zeros = [self.model_char_dict['SENTENCE_PAD']] * self.max_characters_per_token

        for s in sentence_ids:
            s_mask = [True] * len(s)
            while len(s) < max_sentence_len:
                s.append(zeros)
                s_mask.append(False)

            ids.append(s[:max_sentence_len])  # TODO гарантировать, что s и s_mask будут длины max_sentence_len
            mask.append(s_mask[:max_sentence_len])

        ids = torch.LongTensor(ids)
        mask = torch.BoolTensor(mask)
        return ids, mask

    def sentence_to_ids(self, sentence):
        start = [self.model_char_dict['WORD_PAD']] * self.max_characters_per_token
        start[0] = self.model_char_dict['SENTENCE_START']
        end = [self.model_char_dict['WORD_PAD']] * self.max_characters_per_token
        end[0] = self.model_char_dict['SENTENCE_END']

        ids = [start]
        ids += [self.word_to_ids(word) for word in sentence]
        ids += [end]

        return ids

    def word_to_ids(self, word):
        ids = [self.model_char_dict['WORD_START']]
        for c in word:
            ids.append(self.model_char_dict.get(c, self.model_char_dict['UNK']))
        ids.append(self.model_char_dict['WORD_END'])
        ids += [self.model_char_dict['WORD_PAD']] * (self.max_characters_per_token-len(ids))

        # assert len(ids) == self.max_characters_per_token  # TODO разобраться, почему это иногда не срабатывает
        return ids[:self.max_characters_per_token]

    @staticmethod
    def build_char_dict():
        cyrillic = set()
        latin = set()
        punctuation = set('.,;:!?;…‐-‑‒–—―[](){}⟨⟩„“«»“”‘’‹›\'\"&%@#$*№')
        numbers = set('0123456789')

        for i in range(2000):
#            if re.match(r'[a-zA-Z]', chr(i)):
            if re.match(r'[a-z]', chr(i)):
                latin.add(chr(i))
            elif re.match(r'[а-яё]', chr(i)):
#            elif re.match(r'[а-яА-ЯёЁ]', chr(i)):
                cyrillic.add(chr(i))

#        assert len(cyrillic) == 66
#        assert len(latin) == 52

         приводим к списку и сортируем. Это нужно для воспроизводимости
        model_chars = cyrillic | latin | punctuation | numbers
#        model_chars = cyrillic | latin | numbers
        model_chars = list(model_chars)
        model_chars.sort()

        # формируем словарь
        model_char_dict = {key: i for i, key in enumerate([
            'SENTENCE_PAD',
            'WORD_PAD',
            'UNK',
            'WORD_START',
            'WORD_END',
            'SENTENCE_START',
            'SENTENCE_END',
        ])}

        shift = max(model_char_dict.values()) + 1
        model_char_dict.update({c: i + shift for i, c in enumerate(model_chars)})
        return model_char_dict

