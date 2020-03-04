import random

import torch


class Encoder(torch.nn.Module):

    def __init__(self, tagset_size, alphabet_size,
                 in_abet_emb_dim=30, in_tags_emb_dim=20):
        super().__init__()

        self.tagset_size = tagset_size
        self.alphabet_size = alphabet_size

        self.in_abet_emb_dim = in_abet_emb_dim
        self.in_tags_emb_dim = in_tags_emb_dim
        self.hidden_size = self.in_abet_emb_dim + self.in_tags_emb_dim

        self.lem_embedding_layer = torch.nn.Embedding(num_embeddings=alphabet_size,
                                                      embedding_dim=in_abet_emb_dim)

        self.tags_embedding_layer = torch.nn.Embedding(num_embeddings=tagset_size,
                                                       embedding_dim=in_tags_emb_dim)

        self.enc_lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size)

    def forward(self, tags_sequence, lem_chars_sequnece, hidden, batch_size, sequence_length):
        assert tags_sequence.shape[0] == lem_chars_sequnece.shape[0] == sequence_length
        assert tags_sequence.shape[1] == lem_chars_sequnece.shape[1] == batch_size

        # in: seq_len X batch_size; out: seq_len X batch_size X emb_size
        tags_sequence = self.tags_embedding_layer(tags_sequence)
        lem_chars_sequnece = self.lem_embedding_layer(lem_chars_sequnece)

        # in: 2 of seq_len X batch_size X emb_size; out: seq_len X batch_size X emb1_size+emb2_size
        lstm_in = torch.cat((tags_sequence, lem_chars_sequnece), dim=2)

        # in: seq_len X batch_size X hi_size; out: seq_len X batch_size X hi_size
        encoded, hidden = self.enc_lstm(lstm_in, hidden)

        assert encoded.shape[0] == sequence_length and encoded.shape[1] == batch_size

        return encoded, hidden

    def init_hidden(self, batch_size, device):
        hid_a = torch.zeros(1, batch_size, self.hidden_size, device=device)
        hid_b = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return hid_a, hid_b


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dec_lstm = torch.nn.LSTM(output_size, hidden_size)
        self.dec_embedding = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inp_token, hid, batch_size, sequence_length):
        # in: batch_size X lstm_in_size; out: 1 X batch_size X lstm_in_size
        inp_sequence = inp_token.view(1, batch_size, self.output_size)

        # in: 1 X batch_size X lstm_in_size; out: (1 X) batch_size X lstm_out_size
        lstm_out_sequence, hid = self.dec_lstm(inp_sequence, hid)
        lstm_out_token = lstm_out_sequence.view(batch_size, lstm_out_sequence.shape[2])

        # in: batch_size X lstm_out_size; out: batch_size X decoder_out_size
        out_emb = self.dec_embedding(lstm_out_token)

        return out_emb, hid

    def init_hidden(self, batch_size, device):
        hid_a = torch.zeros(1, batch_size, self.hidden_size, device=device)
        hid_b = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return hid_a, hid_b


class EncoderDecoder(torch.nn.Module):
    def __init__(self, decoder_sos_token_index, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.decoder_sos_token_index = decoder_sos_token_index

    def forward(self, source_tags_sequence, source_lems_sequence, target_sequence,
                batch_size, in_sequence_length, out_sequence_length, device, teacher_forcing_ratio=0.5):
        decoder_vocab_size = self.decoder.output_size

        decoder_outputs = torch.zeros(out_sequence_length, batch_size, decoder_vocab_size).to(device)

        encoder_hidden = self.encoder.init_hidden(batch_size, device)
        enc_output, encoder_hidden = self.encoder(source_tags_sequence, source_lems_sequence, encoder_hidden,
                                                  batch_size, in_sequence_length)

        decoder_input_ix = torch.full((batch_size,), self.decoder_sos_token_index).to(torch.int64)
        decoder_input = torch.nn.functional.one_hot(decoder_input_ix,
                                                    decoder_vocab_size).to(torch.float).to(device)

        assert decoder_input.shape[0] == batch_size and decoder_input.shape[1] == decoder_vocab_size

        decoder_hidden = encoder_hidden
        for t in range(out_sequence_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden,
                                                          batch_size, out_sequence_length)

            decoder_outputs[t] = decoder_output
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.max(1)[1]
            top1ohe = torch.nn.functional.one_hot(top1, decoder_vocab_size).to(torch.float).to(device)
            decoder_input = (target_sequence[t] if use_teacher_force else top1ohe)

        return decoder_outputs
