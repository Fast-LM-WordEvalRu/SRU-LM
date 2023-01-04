import torch
from .unidirectional_lm import UnidirectionalLM


class BidirectionalLM(torch.nn.Module):
    def __init__(self, sru=True, char_embedder_params=None, model_params=None):
        super().__init__()
        self.forward_lm = UnidirectionalLM(
            sru=sru,
            char_embedder_params=char_embedder_params,
            model_params=model_params
        )
        self.backward_lm = UnidirectionalLM(
            char_embedder=False,  # костыль, чтобы не инициализовать
            sru=sru,
            char_embedder_params=char_embedder_params,
            model_params=model_params
        )

    def forward(self, ids, mask, return_embeddings=False, **kwargs):
        encoded_chars = self.forward_lm.char_embedder(ids)
        forward_out = self.forward_lm(ids, mask, encoded_chars)
        index = torch.LongTensor(
            [list(range(s, -1, -1)) + list(range(s + 1, encoded_chars.shape[1])) for s in mask.sum(1)]).unsqueeze(2)
        index = index.type_as(ids)
        backward_encoded_chars = torch.take_along_dim(encoded_chars, index, dim=1)
        backward_out = self.backward_lm(ids, mask, backward_encoded_chars)
        backward_out = torch.take_along_dim(backward_out, index, dim=1)

        if return_embeddings:
            return torch.cat([forward_out, backward_out], dim=2)
        else:
            return {
                'forward_out': forward_out,
                'backward_out': backward_out
                }
