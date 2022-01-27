import torch
from .unidirectional_lm import UnidirectionalLM


class BidirectionalLM(torch.nn.Module):
    def __init__(self, sru=True, char_embedder_params=None, model_params=None):
        super().__init__()
        self.use_gpu = False  # TODO: сделать нормальную передачу вызова методов вроде .cuda() в родительский класс

        self.forward_lm = UnidirectionalLM(
            sru=sru,
            char_embedder_params=char_embedder_params,
            model_params=model_params
        )

        # char_embedder = self.forward_lm.char_embedder
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
        backward_encoded_chars = torch.take_along_dim(encoded_chars, index.to(encoded_chars.device), dim=1)
        backward_out = self.backward_lm(ids, mask, backward_encoded_chars)
        backward_out = torch.take_along_dim(backward_out, index, dim=1)

        if return_embeddings:
            return torch.cat([forward_out, backward_out], dim=2)
        else:
            return {
                'forward_out': forward_out,
                'backward_out': backward_out
                }

    def cuda(self):
        self.use_gpu = True
        self.forward_lm.cuda()
        self.backward_lm.cuda()
        return self

    def cpu(self):
        self.use_gpu = False
        self.forward_lm.cpu()
        self.backward_lm.cpu()
        return self
