from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..core import batch_to_ids
from sru_lm.config import batch_size as default_batch_size


class SequencePadder:
    def __init__(self, padding_value, bidirectional):
        self.bidirectional = bidirectional
        self.padding_value = padding_value
    
    def __call__(self, batch):
        texts = [b['text'] for b in batch]
        word_indices = [b['word_indices'] for b in batch]

        text_ids, mask = batch_to_ids(texts)
        padded_word_indices = pad_sequence(word_indices, padding_value=self.padding_value, batch_first=True)
        forward_target = padded_word_indices.roll(-1, [1])
        forward_target[:, -1] = forward_target[:, -2]

        out_dict = {
            'ids': text_ids,
            'mask': mask,
            'forward_target': forward_target
        }

        if self.bidirectional:
            backward_target = padded_word_indices.roll(1, [1])
            out_dict['backward_target'] = backward_target
        
        return out_dict



def get_dataloader(dataset, word_dict, batch_size, num_workers, pin_memory, bidirectional=False):
    if batch_size is None:
        batch_size = default_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                            collate_fn=SequencePadder(padding_value=word_dict['<PAD>'], bidirectional=bidirectional))

    return dataloader
