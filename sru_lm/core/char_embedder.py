import torch
import torch.nn as nn

from .highway import Highway


class CharEmbedder(nn.Module):
    def __init__(self, char_embedding_dim, max_characters_per_token, n_characters, cnn_options, n_highway, output_dim):
        super(CharEmbedder, self).__init__()

        self.char_embedding = nn.Embedding(num_embeddings=n_characters, embedding_dim=char_embedding_dim)

        for i, option in enumerate(cnn_options):
            in_channels = char_embedding_dim
            out_channels = option[1]
            kernel_size = option[0]
            setattr(self, f'char_cnn_{i}', nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size))

        self.n_char_cnn = i + 1

        n_filters = sum(f[1] for f in cnn_options)
        self.highway = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        self.projection = nn.Linear(n_filters, output_dim, bias=True)

        self.max_characters_per_token = max_characters_per_token

    def forward(self, batch_ids):
        x = self.char_embedding(batch_ids.view(-1, self.max_characters_per_token))
        x = torch.transpose(x, 1, 2)

        convs = []
        for i in range(self.n_char_cnn):
            conv = getattr(self, f'char_cnn_{i}')
            convolved = conv(x)
            # (batch_size * sequence_length, n_filters for this width)  [64 * 30, 10, 128]
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = nn.functional.relu(convolved)
            convs.append(convolved)

        token_embedding = torch.cat(convs, dim=-1)
        token_embedding = self.highway(token_embedding)
        token_embedding = self.projection(token_embedding)

        batch_size, sequence_length, _ = batch_ids.size()

        token_embedding = token_embedding.view(batch_size, sequence_length, -1)
        return token_embedding
