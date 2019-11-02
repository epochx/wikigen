import torch.nn as nn
import torch

from .layers import LSTMEncoder
from .utils import gather_last, mean_pooling, max_pooling


class Classifier(nn.Module):
    def __init__(
        self,
        encoder_embeddings,
        encoder_hidden_size,
        num_classes,
        encoder_num_layers=1,
        encoder_input_dropout=0,
        encoder_dropout=0,
        encoder_output_dropout=0,
        bidirectional=True,
        aggregator="mean",
        tag_embeddings=None,
        message_embeddings=None,
        message_encoder_hidden_size=None,
    ):

        super(Classifier, self).__init__()

        self.encoder_embeddings = encoder_embeddings
        self.tag_embeddings = tag_embeddings

        self.encoder_hidden_size = encoder_hidden_size

        self.bidirectional = bidirectional

        self.encoder = LSTMEncoder(
            encoder_embeddings,
            self.encoder_hidden_size,
            num_layers=encoder_num_layers,
            input_dropout=encoder_input_dropout,
            dropout=encoder_dropout,
            output_dropout=encoder_output_dropout,
            tag_embeddings=self.tag_embeddings,
        )

        self.message_embeddings = None
        self.message_encoder_hidden_size = None

        self.aggregator = aggregator

        self.num_classes = num_classes

        classifier_input_size = self.encoder_hidden_size
        if self.bidirectional:
            classifier_input_size += self.encoder_hidden_size

        if (
            message_embeddings is not None
            and message_encoder_hidden_size is not None
        ):
            self.message_embeddings = message_embeddings
            self.message_encoder_hidden_size = message_encoder_hidden_size
            self.message_encoder = LSTMEncoder(
                message_embeddings,
                self.message_encoder_hidden_size,
                input_dropout=encoder_input_dropout,
                output_dropout=encoder_output_dropout,
                bidirectional=True,
            )

            # message encoder is always bidirectional
            classifier_input_size += 2 * message_encoder_hidden_size

        self.classification_layer = nn.Linear(
            classifier_input_size, self.num_classes
        )

        self.class_loss_function = nn.KLDivLoss(reduce="batchmean")

    def forward(
        self, src_batch_tuple, tgt_batch_tuple, src_message_batch_tuple=None
    ):

        src_batch_sequences = src_batch_tuple.sequences
        src_batch_tag_sequences = src_batch_tuple.tag_sequences
        src_batch_lengths = src_batch_tuple.lengths

        h_t, _ = self.encode(
            src_batch_sequences, src_batch_lengths, src_batch_tag_sequences
        )

        if src_message_batch_tuple is not None:
            assert self.message_encoder is not None
            src_message_batch_sequences = src_message_batch_tuple.sequences
            src_message_batch_lengths = src_message_batch_tuple.lengths

            src_message_encoder_hidden_states = self.message_encoder(
                src_message_batch_sequences, src_message_batch_lengths
            )

            message_h_t = mean_pooling(
                src_message_encoder_hidden_states, src_message_batch_lengths
            )

            h_t = torch.cat([h_t, message_h_t], 1)

        # batch_size, hidden_x_dirs -> batch_size, num_classes
        class_logits = self.classification_layer(h_t)

        class_probs = nn.functional.softmax(class_logits, 1)

        class_log_probs = nn.functional.log_softmax(class_logits, 1)

        class_labels = tgt_batch_tuple.classes
        class_loss = self.class_loss_function(class_log_probs, class_labels)

        return class_loss, class_probs

    def encode(
        self, src_batch_sequences, src_batch_lengths, src_batch_tag_sequences
    ):
        # Encode source code, comments and code/docstring if present
        # batch_size, seq_len, hidden_x_dirs
        encoder_hidden_states = self.encoder.forward(
            src_batch_sequences, src_batch_lengths, src_batch_tag_sequences
        )

        if self.aggregator == "mean":
            # batch_size, hidden_x_dirs
            h_t = mean_pooling(encoder_hidden_states, src_batch_lengths)

        elif self.aggregator == "max":
            # batch_size, hidden_x_dirs
            h_t = max_pooling(encoder_hidden_states)

        elif self.aggregator == "last":
            # batch_size, hidden_x_dirs
            h_t = gather_last(
                encoder_hidden_states,
                src_batch_lengths,
                bidirectional=self.bidirectional,
            )
        else:
            raise NotImplementedError

        return h_t, encoder_hidden_states


class Bottle(nn.Module):
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.contiguous().view(size[0], size[1], -1)


class BottleLinear(Bottle, nn.Linear):
    pass
