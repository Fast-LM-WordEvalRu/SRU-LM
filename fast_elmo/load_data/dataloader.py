from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..core import batch_to_ids
from fast_elmo.config import batch_size


class SequencePadder:
    def __init__(self, word_dict, bidirectional=False):
        self.word_dict = word_dict
        self.bidirectional = bidirectional

    def process_batch_single_direction(self, batch, direction='forward'):
        text_batch = [b[f'{direction}_text'] for b in batch]
        ids, mask = batch_to_ids(text_batch)
        targets = [b[f'{direction}_target'] for b in batch]
        targets = pad_sequence(targets, padding_value=self.word_dict['<PAD>'], batch_first=True)
        return ids, mask, targets

    def __call__(self, batch):
        out_dict = {}

        directions = ['forward']
        if self.bidirectional:
            directions.append('backward')

        for direction in directions:
            ids, mask, targets = self.process_batch_single_direction(batch, direction)
            out_dict[f'{direction}_ids'] = ids
            out_dict[f'{direction}_mask'] = mask
            out_dict[f'{direction}_targets'] = targets

        return out_dict


def get_dataloader(dataset, word_dict):
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4,
                            collate_fn=SequencePadder(word_dict))

    return dataloader
