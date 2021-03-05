from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..core import batch_to_ids
from sru_lm.config import batch_size as default_batch_size


class SequencePadder:
    def __init__(self, padding_value, bidirectional, model_chars, max_character_per_token):
        self.bidirectional = bidirectional
        self.padding_value = padding_value
        self.model_chars = model_chars
        self.max_character_per_token = max_character_per_token
    
    def __call__(self, batch):
        texts = [b['text'] for b in batch]
        word_indices = [b['word_indices'] for b in batch]

        text_ids, mask = batch_to_ids(texts, self.model_chars, self.max_character_per_token)
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


def get_dataloader(dataset, word_dict, bidirectional=False, batch_size=None, model_chars=None, max_character_per_token=None):
    if batch_size is None:
        batch_size = default_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4,
                            collate_fn=SequencePadder(padding_value=word_dict['<PAD>'],
                                                      bidirectional=bidirectional,
                                                      model_chars=model_chars,
                                                      max_character_per_token=max_character_per_token))

    return dataloader
