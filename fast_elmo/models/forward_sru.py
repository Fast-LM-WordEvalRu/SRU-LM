from torch import nn
from sru import SRU  # TODO: использовать SRUCell и конструировать языковую модель вручную

from fast_elmo.core.char_embedder import CharEmbedder
from fast_elmo.config import char_embedder_params, sru_model_params, batch_size


class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        # пока все параметры задаю через глобальные переменные, потом это поправлю

        self.char_embedder = CharEmbedder(**char_embedder_params)

        self.forward_LM = SRU(
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

        hidden = next(self.forward_LM.parameters()).data.new(sru_model_params['n_sru_layers'],
                                                             batch_size, sru_model_params['output_dim']).zero_()
        hidden = hidden.cuda()

        lm_out, hidden = self.forward_LM(encoded_chars, hidden, mask_pad=inverted_mask)

        return lm_out.permute(1, 0, 2)
