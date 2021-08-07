import torch
from torch import nn
from ..layers.fastspeech import FFTransformer, DurationPredictor


class FastSpeech(nn.Module):
    def __init__(
        self,
        vocab_size=32,
        n_mels=80,
        embedding_size=512,
        n_layers=3,
        attention_num_heads=8,
        transformer_conv_kernel_size=3,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        layers_dropout=0.1,
        duration_predictor_filter_size=256,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = FFTransformer(
            n_layers,
            embedding_size,
            attention_num_heads,
            transformer_conv_kernel_size,
            embedding_dropout,
            attention_dropout,
            layers_dropout,
        )

        self.decoder = FFTransformer(
            n_layers,
            embedding_size,
            attention_num_heads,
            transformer_conv_kernel_size,
            embedding_dropout,
            attention_dropout,
            layers_dropout,
        )

        self.duration_predictor = DurationPredictor(
            embedding_size,
            duration_predictor_filter_size,
            duration_predictor_kernel_size,
            duration_predictor_dropout,
        )

        self.linear = nn.Linear(embedding_size, n_mels)

    @staticmethod
    def regulate_lenghts(input_seq, durations):
        max_mel_lenght = torch.max(torch.sum(durations, dim=1)).item()
        alignments = torch.zeros(durations.shape[0], max_mel_lenght, durations.shape[1])

        for i in range(durations.shape[0]):
            token_start_position = 0
            for j in range(durations.shape[1]):
                token_duration = durations[i][j]
                alignments[
                    i, token_start_position : token_start_position + token_duration, j
                ] = 1.0
                token_start_position += token_duration

        output_sequence = torch.bmm(alignments, input_seq)

        return output_sequence, alignments

    def forward(self, input_tokens, target_durations, encoder_mask, decoder_mask):
        embeddings = self.embedding(input_tokens)
        encoder_out = self.encoder(embeddings, encoder_mask)
        durations = self.duration_predictor(encoder_out, encoder_mask)

        encoder_out_expanded, alignments = self.regulate_lenghts(
            encoder_out, target_durations
        )
        decoder_output = self.decoder(encoder_out_expanded, decoder_mask)
        output_mels = self.linear(decoder_output)

        return output_mels, durations, alignments

    def inference(self, input_tokens, encoder_mask):
        embeddings = self.embedding(input_tokens)
        encoder_out = self.encoder(embeddings, encoder_mask)
        durations = self.duration_predictor(encoder_out, encoder_mask)

        durations = torch.exp(durations).round().long()
        encoder_out_expanded, alignments = self.regulate_lenghts(encoder_out, durations)

        output_mels_lenghts = torch.sum(durations, dim=1)
        max_mel_lenght = torch.max(output_mels_lenghts).item()

        decoder_mask = torch.zeros(
            encoder_mask.shape[0], max_mel_lenght, dtype=torch.bool
        )
        for i, mel_lenght in enumerate(output_mels_lenghts):
            decoder_mask[i][mel_lenght:] = 1

        decoder_output = self.decoder(encoder_out_expanded, decoder_mask)
        output_mels = self.linear(decoder_output)

        return output_mels, alignments, decoder_mask
