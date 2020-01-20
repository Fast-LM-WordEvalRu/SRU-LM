#   Author: Artem Skiba
#   Created: 20/01/2020

import sys

import sru
import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim=1024, dropout=0.5,
                 rnn_dropout=0.1, depth=2, bias=-3, trainable=False):
        """
        num_embeddings -- size of Embedding Layer input (num of words)
        embedding_dim -- size of Embedding Layer output (n_d)
        dropout -- % of neurons to disable in dropout layer
        rnn_dropout -- % of neurons to disable in rnn layer (HAHA NO!)
        depth -- number of rnn layers
        bias -- I don't know what is it
        """
        super(Model, self).__init__()

        self.embedding_dim = embedding_dim
        self.depth = depth
        self.bias = bias
        self.trainable = trainable

        self.relu = nn.ReLU()

        self.rnn = sru.SRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=depth,
            dropout=dropout,
            rnn_dropout=rnn_dropout,
            bidirectional=True,
            rescale=False,  # wtf is rescale?
            v1=True,  # wtf is v1?
            highway_bias=bias)

        if trainable:
            self.output_layer = nn.Linear(embedding_dim, vocabulary_size)

        self.drop = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        val_range = (3.0 / self.embedding_dim) ** 0.5

        params = list(self.output_layer.parameters())
        for p in params:
            if p.dim() > 1:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = Variable(weight.new(self.depth, batch_size, self.embedding_dim * 2).zero_())
        return zeros

    def print_pnorm(self):
        norms = ["{:.0f}".format(x.norm().item()) for x in self.parameters()]
        sys.stdout.write("\tp_norm: {}\n".format(
            norms
        ))

    def forward(self, emb, hidden, mask):
        emb = emb.permute(1, 0, 2)
        output, hidden = self.rnn(emb, hidden, mask_pad=1 - mask.T)
        output = self.drop(output)
        out_forward, out_backward = torch.split(output, self.embedding_dim, 2)

        if self.trainable:
            out_forward = self.output_layer(out_forward)
            out_backward = self.output_layer(out_backward)

        return out_forward, out_backward, hidden
