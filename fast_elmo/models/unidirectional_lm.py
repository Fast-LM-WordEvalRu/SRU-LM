from torch import nn
from sru import SRU

from fast_elmo.core.char_embedder import CharEmbedder
from fast_elmo.config import char_embedder_params, sru_model_params, batch_size


class UnidirectionalLM(nn.Module):
    def __init__(self, char_embedder=None):
        super().__init__()
        # пока все параметры задаю через глобальные переменные, потом это поправлю

        if char_embedder is None:
            self.char_embedder = CharEmbedder(**char_embedder_params)
        else:
            self.char_embedder = char_embedder

        self.use_gpu = False

        self._language_model = SRU(
            input_size=sru_model_params['output_dim'],
            hidden_size=sru_model_params['output_dim'],
            use_tanh=True,
            num_layers=sru_model_params['n_sru_layers']
        )

    def forward(self, ids, mask):
        encoded_chars = self.char_embedder(ids)
        encoded_chars = encoded_chars.permute(1, 0, 2)
        inverted_mask = ~mask
        inverted_mask = inverted_mask.T

        hidden = next(self._language_model.parameters()).data.new(sru_model_params['n_sru_layers'],
                                                                  batch_size, sru_model_params['output_dim']).zero_()
        if self.use_gpu:
            hidden = hidden.cuda()

        lm_out, hidden = self._language_model(encoded_chars, hidden, mask_pad=inverted_mask)

        return lm_out.permute(1, 0, 2)

    def cuda(self):
        self.use_gpu = True
        super().cuda()
        return self

    def cpu(self):
        self.use_gpu = False
        super().cpu()
        return self

