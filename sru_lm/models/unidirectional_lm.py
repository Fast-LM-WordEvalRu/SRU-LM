from torch import nn
from sru import SRU

from sru_lm.core.char_embedder import CharEmbedder
from sru_lm.config import (
    char_embedder_params as default_char_embedder_params,
    sru_model_params as default_sru_model_params,
    lstm_model_params as default_lstm_model_params
)


class UnidirectionalLM(nn.Module):
    def __init__(self, char_embedder=None, sru=True, char_embedder_params=None, model_params=None):
        super().__init__()

        if char_embedder is None:
            if char_embedder_params is None:
                char_embedder_params = default_char_embedder_params
            self.char_embedder = CharEmbedder(**char_embedder_params)
        else:
            self.char_embedder = char_embedder

        self.use_gpu = False
        self.sru = sru

        if self.sru:
            if model_params is None:
                model_params = default_sru_model_params
            self._language_model = SRU(
                input_size=model_params['output_dim'],
                hidden_size=model_params['output_dim'],
                use_tanh=True,
                num_layers=model_params['n_layers']
            )
        else:
            if model_params is None:
                model_params = default_lstm_model_params
            self._language_model = nn.LSTM(
                input_size=model_params['output_dim'],
                hidden_size=model_params['output_dim'],
                num_layers=model_params['n_layers'],
                batch_first=True
            )

    def forward(self, ids, mask):
        batch_size = ids.shape[0]
        if self.sru:
            encoded_chars = self.char_embedder(ids)
            encoded_chars = encoded_chars.permute(1, 0, 2)
            inverted_mask = ~mask
            inverted_mask = inverted_mask.T

            hidden = next(self._language_model.parameters()).data.new(sru_model_params['n_layers'],
                                                                      batch_size, sru_model_params['output_dim']).zero_()
            if self.use_gpu:
                hidden = hidden.cuda()

            lm_out, hidden = self._language_model(encoded_chars, hidden, mask_pad=inverted_mask)

            return lm_out.permute(1, 0, 2)
        else:
            encoded_chars = self.char_embedder(ids)
            # lens = mask.sum(dim=1)
            # encoded_chars = nn.utils.rnn.pack_padded_sequence(encoded_chars, lens, batch_first=True)
            lstm_out, _ = self._language_model(encoded_chars)
            # lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            return lstm_out

    def cuda(self):
        self.use_gpu = True
        super().cuda()
        return self

    def cpu(self):
        self.use_gpu = False
        super().cpu()
        return self

