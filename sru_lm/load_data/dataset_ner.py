import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, path_to_conll_file, ner_labels):
        texts = []
        targets = []
        labels_dict = {k: v for v, k in enumerate(ner_labels)}

        with open(path_to_conll_file) as f:
            text = []
            text_targets = [labels_dict['O']]
            for line in f.readlines():
                line = line.strip()
                if line:
                    splitted_line = line.split()
                    word = [w for w in splitted_line[0].lower() if w.isalnum()]

                    if len(word) > 0:
                        word = ''.join(word)
                        text.append(word)
                        text_targets.append(labels_dict.get(splitted_line[-1], labels_dict['O']))
                else:
                    text_targets += [labels_dict['O'], labels_dict['O']]  # очень плохая практика. Моя функция batch_to_ids увеличивает размер текста на 3, приходится это учитывать везде, где она используется
                    texts.append(text)
                    targets.append(text_targets)
                    text = []
                    text_targets = [labels_dict['O']]

        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            'text': self.texts[item],
            'target': torch.LongTensor(self.targets[item])
        }
