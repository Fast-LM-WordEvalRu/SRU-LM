import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, path_to_conll_file):
        texts = []
        targets = []

        labels_dict = {
             'O': 0,
             'B-Loc': 1,
             'I-Loc': 2,
             'B-LocOrg': 3,
             'I-LocOrg': 4,
             'B-Location': 5,
             'I-Location': 6,
             'B-Org': 7,
             'I-Org': 8,
             'B-Person': 9,
             # 'I-Facility': 10,
             'I-Person': 10,
        }

        with open(path_to_conll_file) as f:
            text = []
            text_targets = [0]
            for line in f.readlines():
                line = line.strip()
                if line:
                    splitted_line = line.split()
                    text.append(splitted_line[0])
                    text_targets.append(labels_dict.get(splitted_line[-1], 0))
                else:
                    texts.append(text)
                    text_targets.append(0)
                    targets.append(text_targets)
                    text = []
                    text_targets = [0]

        texts = texts[1:]
        targets = targets[1:]
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            'raw_text': self.texts[item],
            'target': torch.LongTensor(self.targets[item])
        }
