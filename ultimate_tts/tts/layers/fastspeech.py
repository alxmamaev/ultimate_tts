import torch
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        assert embedding_size % 2 == 0, "embedding_size must be divisible by 2"

        self.embedding_size = embedding_size
        inverse_frequencies = 1 / (
            10000 ** (torch.arange(0.0, embedding_size, 2.0) / embedding_size)
        )
        self.register_buffer("inverse_frequencies", inverse_frequencies)

    def forward(self, positions):
        position_frequencies = torch.outer(positions, self.inverse_frequencies)
        position_embedding = torch.cat(
            [position_frequencies.sin(), position_frequencies.cos()], axis=1
        )

        return position_embedding


class FFTBlock(nn.Module):
    def __init__(
        self,
        embedding_size,
        attention_num_heads,
        conv_kernel_size,
        attention_dropout,
        layers_dropout,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embedding_size, attention_num_heads, attention_dropout
        )
        self.dropout_attention_leayer = nn.Dropout(layers_dropout)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                embedding_size,
                embedding_size,
                conv_kernel_size,
                1,
                (conv_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                embedding_size,
                embedding_size,
                conv_kernel_size,
                1,
                (conv_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(layers_dropout),
        )

        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)

    def forward(self, X, mask):
        attention_input = X.permute(1, 0, 2)
        attn_output, attn_output_weights = self.self_attn(
            attention_input, attention_input, attention_input, key_padding_mask=mask
        )
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = self.dropout_attention_leayer(attn_output)
        attn_output = self.layer_norm_1(attn_output + X)

        attn_output.data.masked_fill_(mask.unsqueeze(2), 0.0)

        conv_output = self.conv_layers(attn_output.transpose(1, 2)).transpose(1, 2)
        conv_output = self.layer_norm_2(conv_output + attn_output)

        conv_output.data.masked_fill_(mask.unsqueeze(2), 0.0)

        return conv_output


class FFTransformer(nn.Module):
    def __init__(
        self,
        n_layers,
        embedding_size,
        attention_num_heads,
        conv_kernel_size,
        embedding_dropout,
        attention_dropout,
        layers_dropout,
    ):
        super().__init__()
        self.positional_encoder = PositionalEncoder(embedding_size)
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(
                FFTBlock(
                    embedding_size,
                    attention_num_heads,
                    conv_kernel_size,
                    attention_dropout,
                    layers_dropout,
                )
            )

        self.dropout_embedding = nn.Dropout(embedding_dropout)
        self.attention_num_heads = attention_num_heads

    def forward(self, X, mask):
        positions = torch.arange(X.shape[1], device=X.device, dtype=X.dtype)
        position_embedding = self.positional_encoder(positions)
        position_embedding = position_embedding.unsqueeze(0)
        out = self.dropout_embedding(X + position_embedding)

        for layer in self.layers:
            out = layer(out, mask)

        return out


class DurationPredictor(nn.Module):
    def __init__(self, encoder_embedding_size, filter_size, kernel_size, dropout):
        super().__init__()

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    encoder_embedding_size,
                    filter_size,
                    kernel_size,
                    1,
                    (kernel_size - 1) // 2,
                ),
                nn.Conv1d(
                    filter_size, filter_size, kernel_size, 1, (kernel_size - 1) // 2
                ),
            ]
        )

        self.norm_layers = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(filter_size), nn.ReLU(), nn.Dropout(dropout))
                for i in range(2)
            ]
        )

        self.linear = nn.Sequential(nn.Linear(filter_size, 1), nn.ReLU())

    def forward(self, X, mask):
        out = X
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            out = out.transpose(1, 2)
            out = conv(out)
            out = out.transpose(1, 2)
            out = norm(out)

        out = self.linear(out)
        out = out.squeeze(2)
        out.data.masked_fill_(mask, 0.0)

        return out
