import random
import torch
import torch.nn as nn

from .attention import AttentionLayer
from .layers import LSTMEncoder
from .utils import gather_last, mean_pooling, max_pooling


class BOWEncoder(nn.Module):
    def __init__(self, embeddings):
        super(BOWEncoder, self).__init__()

        self.embeddings = embeddings

    def forward(self, sequences, sequence_legths, **kwargs):
        # We make this return None to make it compatible with the LSTM encoder
        return self.embeddings(sequences)


class Seq2seq(nn.Module):
    def __init__(
        self,
        encoder_embeddings,
        decoder_embeddings,
        weights,
        encoder,
        decoder_hidden_size,
        encoder_num_layers=1,
        encoder_input_dropout=0,
        encoder_dropout=0,
        encoder_output_dropout=0,
        decoder_input_dropout=0,
        decoder_output_dropout=0,
        encoder_hidden_size=200,
        bidirectional=True,
        aggregator=None,
        teacher_forcing_p=0.5,
        attention="dot",
        num_classes=None,
        tag_embeddings=None,
    ):

        super(Seq2seq, self).__init__()

        if encoder not in ["bow", "lstm", "linear"]:
            raise NotImplementedError

        self.encoder_embeddings = encoder_embeddings
        self.decoder_embeddings = decoder_embeddings
        self.tag_embeddings = tag_embeddings

        self.decoder_input_dropout = nn.Dropout(decoder_input_dropout)
        self.decoder_output_dropout = nn.Dropout(decoder_output_dropout)

        self.encoder_hidden_size = encoder_hidden_size

        self.encoder_type = encoder
        self.bidirectional = False

        if self.encoder_type == "bow":
            self.encoder = BOWEncoder(encoder_embeddings)

        elif self.encoder_type == "linear":
            encoder_emb_dim = self.encoder_embeddings.embedding_dim
            if self.tag_embeddings is not None:
                encoder_emb_dim += self.tag_embeddings.embedding_dim

            self.encoder = BottleLinear(encoder_emb_dim, encoder_hidden_size)

        elif self.encoder_type == "lstm":
            self.bidirectional = bidirectional
            self.encoder = LSTMEncoder(
                encoder_embeddings,
                self.encoder_hidden_size,
                num_layers=encoder_num_layers,
                dropout=encoder_dropout,
                input_dropout=encoder_input_dropout,
                output_dropout=encoder_output_dropout,
                tag_embeddings=self.tag_embeddings,
            )

        self.aggregator = aggregator

        self.decoder_hidden_size = decoder_hidden_size

        self.decoder = nn.LSTM(
            self.decoder_embeddings.embedding_dim,
            self.decoder_hidden_size,
            batch_first=True,
        )

        self.teacher_forcing_p = teacher_forcing_p

        self.attention_layer = AttentionLayer(
            self.decoder_hidden_size, scoring_scheme=attention
        )

        # The 2 appears because we will concatenate the decoded vector with the
        # attended decoded vector
        num_catted_vectors = 2

        self.composer_layer = nn.Linear(
            self.decoder_hidden_size * num_catted_vectors,
            self.decoder_hidden_size,
        )

        self.output_layer = nn.Linear(
            self.decoder_hidden_size, self.decoder_embeddings.num_embeddings
        )

        self.loss_function = nn.CrossEntropyLoss(weight=weights)

        self.num_classes = num_classes
        if self.num_classes is not None:
            self.classification_layer = nn.Linear(
                self.decoder_hidden_size, self.num_classes
            )

            self.class_loss_function = nn.KLDivLoss(reduce="batchmean")

    def forward(self, src_batch_tuple, tgt_batch_tuple):

        class_name = self.__class__.__name__

        src_batch_sequences = src_batch_tuple.sequences
        src_batch_tag_sequences = src_batch_tuple.tag_sequences
        src_batch_lengths = src_batch_tuple.lengths
        src_batch_mask = src_batch_tuple.masks

        # tgt batch includes BOS and EOS
        tgt_batch_sequences = tgt_batch_tuple.sequences
        batch_size, tgt_seq_len = tgt_batch_sequences.size()

        h_t, encoder_hidden_states = self.encode(
            src_batch_sequences, src_batch_lengths, src_batch_tag_sequences
        )

        logits = []
        predictions = []
        attention = []

        # batch_size, 1
        tgt_batch_sequences_i = tgt_batch_sequences[:, 0].unsqueeze(1)

        # FIXME: Assumes encoder output hidden dimension (after layers and
        # bidirectionality) equals the decoder hidden size

        # 1, batch_size, hidden_x_dirs
        decoder_hidden_tuple_i = (h_t.unsqueeze(0), h_t.unsqueeze(0))

        # teacher forcing p
        p = random.random()

        self.attention = []

        # we skip the EOS as input for the decoder
        for i in range(tgt_seq_len - 1):

            decoder_hidden_tuple_i, logits_i = self.generate(
                tgt_batch_sequences_i,
                decoder_hidden_tuple_i,
                encoder_hidden_states,
                src_batch_mask,
            )

            # batch_size
            _, predictions_i = logits_i.max(1)

            logits.append(logits_i)
            predictions.append(predictions_i)

            if self.training and p <= self.teacher_forcing_p:
                # batch_size, 1
                tgt_batch_sequences_i = tgt_batch_sequences[:, i + 1].unsqueeze(1)
            else:
                # batch_size, 1
                tgt_batch_sequences_i = predictions_i.unsqueeze(1)
                if src_batch_sequences.is_cuda:
                    tgt_batch_sequences_i = tgt_batch_sequences_i.cuda()

        # (seq_len, batch_size)
        predictions = torch.stack(predictions, 0)

        # (batch_size, seq_len)
        predictions = predictions.t().contiguous()

        # (seq_len, batch_size, output_size)
        logits = torch.stack(logits, 0)

        # (batch_size, seq_len, output_size)
        logits = logits.transpose(0, 1).contiguous()

        # (batch_size*seq_len, output_size)
        flat_logits = logits.view(batch_size * (tgt_seq_len - 1), -1)

        # (batch_size, seq_len)
        labels = tgt_batch_sequences[:, 1:].contiguous()

        # (batch_size*seq_len)
        flat_labels = labels.view(-1)

        loss = self.loss_function(flat_logits, flat_labels)

        if self.num_classes is not None:
            # batch_size, hidden_x_dirs -> batch_size, num_classes
            class_logits = self.classification_layer(h_t)

            class_probs = nn.functional.softmax(class_logits, 1)

            class_log_probs = nn.functional.log_softmax(class_logits, 1)

            class_labels = tgt_batch_tuple.classes
            class_loss = self.class_loss_function(class_log_probs, class_labels)
        else:
            class_loss = None
            class_probs = None

        return loss, predictions, class_loss, class_probs

    def generate(
        self,
        tgt_batch_sequences_i,
        decoder_hidden_tuple_i,
        encoder_hidden_states,
        src_batch_mask,
    ):
        """

        :param tgt_batch_i: torch.LongTensor(1, batch_size)
        :param decoder_hidden_tuple_i: tuple(torch.FloatTensor(1, batch_size, hidden_size))
        :param encoder_hidden_states: torch.FloatTensor(batch_size, seq_len, hidden_x_dirs)
        :param src_batch_mask: torch.LongTensor(batch_size, seq_len)
        :param comment_hidden_states: ?
        :param com_batch_mask: ?
        :return:
        """

        # (batch_size, 1, embedding_size)
        emb_tgt_batch_i = self.decoder_embeddings(tgt_batch_sequences_i)
        emb_tgt_batch_i = self.decoder_input_dropout(emb_tgt_batch_i)

        # (batch_size, 1, hidden_x_dirs) and (1, batch_size, hidden_size)
        decoder_hidden_states_i, decoder_hidden_tuple_i = self.decoder(
            emb_tgt_batch_i, decoder_hidden_tuple_i
        )

        # batch_size, hidden_x_dirs
        s_i = decoder_hidden_states_i.squeeze(1)

        # (batch_size, hidden_x_dirs) and  (batch_size, seq_len)
        t_i, attn_i = self.attention_layer.forward(
            encoder_hidden_states, s_i, src_batch_mask
        )

        self.attention.append(attn_i)

        # batch_size, 2*hidden_x_dirs
        s_t_i = torch.cat([s_i, t_i], 1)

        # batch_size, hidden_x_dirs
        new_s_i = self.composer_layer(s_t_i)
        new_s_i = self.decoder_output_dropout(new_s_i)

        # batch_size, output_size
        logits_i = self.output_layer(new_s_i)

        return decoder_hidden_tuple_i, logits_i

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
