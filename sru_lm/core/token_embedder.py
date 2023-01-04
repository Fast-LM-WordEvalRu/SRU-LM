import torch
import torch.nn as nn


class TokenEmbedder(nn.Module):
    def __init__(self, emb_dim, vocab_size):
        super(TokenEmbedder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

    def forward(self, batch_ids):
        return self.embedding(batch_ids)
