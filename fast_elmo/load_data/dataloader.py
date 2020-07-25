from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..core import batch_to_ids
from fast_elmo.config import batch_size


class SequencePadder:
    def __init__(self, word_dict):
        self.word_dict = word_dict

    def __call__(self, batch):
        text_batch = [b['raw_text'] for b in batch]
        ids, mask = batch_to_ids(text_batch)
        target_key = 'target' if 'target' in batch[0].keys() else 'forward_target'
        targets = [b[target_key] for b in batch]
        targets = pad_sequence(targets, padding_value=self.word_dict['PAD'], batch_first=True)

        return ids, mask, targets


def get_dataloader(dataset, word_dict):
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4,
                            collate_fn=SequencePadder(word_dict))

    return dataloader
