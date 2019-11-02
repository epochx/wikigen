import torch
import torch.nn as nn

from .utils import (
    pack_rnn_input,
    unpack_rnn_output,
    gather_last,
    feed_forward_rnn,
)


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        embeddings,
        encoding_dim,
        num_layers=1,
        bidirectional=True,
        input_dropout=0.0,
        dropout=0.0,
        output_dropout=0.0,
        tag_embeddings=None,
    ):

        super(LSTMEncoder, self).__init__()

        self.embeddings = embeddings
        self.tag_embeddings = tag_embeddings
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers

        self.input_dropout = nn.Dropout(input_dropout)
        self.dropout = dropout
        self.output_dropout = nn.Dropout(output_dropout)

        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1

        lstm_input_size = self.embeddings.embedding_dim
        if self.tag_embeddings is not None:
            lstm_input_size += self.tag_embeddings.embedding_dim

        self.encoder = torch.nn.LSTM(
            lstm_input_size,
            self.encoding_dim,
            num_layers=self.num_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
        )

    def forward(self, sequences, lengths, tag_sequences=None):

        embedded_batch = self.embeddings(sequences)

        if tag_sequences is not None and self.tag_embeddings is not None:
            embedded_tag_batch = self.tag_embeddings(tag_sequences)
            embedded_batch = torch.cat([embedded_batch, embedded_tag_batch], 2)

        encoded_batch, _ = feed_forward_rnn(
            self.encoder,
            embedded_batch,
            lengths=lengths)

        encoded_batch = self.output_dropout(encoded_batch)

        return encoded_batch
