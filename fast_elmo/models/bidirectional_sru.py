from torch import nn

from .one_directional_sru import OneDirectionalSRUModel


class BidirectionalSRUModel(nn.Module):
    def __init__(self):
        super(BidirectionalSRUModel, self).__init__()

        self.forward_lm = OneDirectionalSRUModel()
        self.backward_lm = OneDirectionalSRUModel()

    def forward(self, forward_ids, forward_mask, backward_ids, backward_mask, concat=False):
        out_dict = {
            'forward_out': self.forward_lm(forward_ids, forward_mask),
            'backward_out': self.backward_lm(backward_ids, backward_mask),
        }

        if concat:
            raise NotImplementedError
        else:
            return out_dict
