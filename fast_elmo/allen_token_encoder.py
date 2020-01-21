import torch
from allennlp.modules.elmo import _ElmoCharacterEncoder, batch_to_ids

class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self._token_embedder = _ElmoCharacterEncoder(options_file, weight_file)        
        
    def forward(self, raw_sentences):
        ids = batch_to_ids(raw_sentences)
        token_embedding = self._token_embedder(ids)
        return token_embedding
    
tmp_model = Model()
tmp_model.forward([['It', 'is', 'first', 'raw', 'sentence', '.'], ['Another', 'one','.']]) # <------ returns dictionary with 'mask' and 'token_embedding' keys