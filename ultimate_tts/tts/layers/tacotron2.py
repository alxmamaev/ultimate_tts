import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.dropout import Dropout


class LocationSensetiveAttention(nn.Module):
    def __init__(
        self, embedding_size, query_size, memory_size, kernel_size, num_filers
    ):
        super().__init__()

        self.query_layer = nn.Linear(query_size, embedding_size)
        self.memory_layer = nn.Linear(memory_size, embedding_size)
        self.location_layer_conv = nn.Conv1d(
            1,
            num_filers,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.location_layer_linear = nn.Linear(num_filers, embedding_size, bias=False)

        self.v = nn.Linear(embedding_size, 1, bias=False)

    def forward(self, query, alignment_state, memory, processed_memory, mask):
        processed_query = self.query_layer(query)
        processed_query = processed_query.unsqueeze(1)

        expanded_alignments = alignment_state.unsqueeze(1)

        processed_location_features = self.location_layer_conv(expanded_alignments)
        processed_location_features = processed_location_features.transpose(1, 2)
        processed_location_features = self.location_layer_linear(
            processed_location_features
        )

        energy = self.v(
            torch.tanh(processed_location_features + processed_query + processed_memory)
        )
        energy = energy.squeeze(2)

        energy.data.masked_fill_(mask, float("-inf"))
        alignments = torch.softmax(energy, 1)

        next_state = alignment_state + alignments

        attention_context = torch.bmm(alignments.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, alignments, next_state


class Postnet(nn.Module):
    def __init__(
        self, n_mels, embedding_size, n_convolutions, kernel_size, dropout_rate
    ):
        super().__init__()

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    n_mels, embedding_size, kernel_size, padding=(kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(embedding_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            )
        )

        for _ in range(n_convolutions - 2):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        embedding_size,
                        embedding_size,
                        kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.BatchNorm1d(embedding_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    embedding_size, n_mels, kernel_size, padding=(kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(n_mels),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            )
        )

    def forward(self, X):
        out = X
        for conv in self.convolutions:
            out = conv(out)

        return out


class Prenet(nn.Module):
    def __init__(self, n_mels, n_layers, embedding_size, dropout_rate):
        super().__init__()

        layers_sizes = [embedding_size for i in range(n_layers)]

        input_sizes = [n_mels] + layers_sizes[:-1]
        output_sizes = layers_sizes

        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()

        for input_size, output_size in zip(input_sizes, output_sizes):
            layer = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
            self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
            # We are using dropout at the test time
            out = torch.dropout(out, p=self.dropout_rate, train=True)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        vocab_embedding_size,
        encoder_embedding_size,
        n_rnn_layers,
        n_convolutions,
        kernel_size,
        dropout_rate,
    ):
        assert (
            encoder_embedding_size % 2 == 0
        ), "encoder_embedding_size must be divisible by 2"

        super().__init__()

        self.embdedding = nn.Embedding(vocab_size, vocab_embedding_size)
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    vocab_embedding_size,
                    encoder_embedding_size,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(encoder_embedding_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            )
        )

        for _ in range(n_convolutions - 1):
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    encoder_embedding_size,
                    encoder_embedding_size,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.BatchNorm1d(encoder_embedding_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            )
            self.convolutions.append(conv_layer)

        self.lstm = nn.LSTM(
            encoder_embedding_size,
            encoder_embedding_size // 2,
            num_layers=n_rnn_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, X):
        out = self.embdedding(X)
        out = out.transpose(2, 1)

        for conv in self.convolutions:
            out = conv(out)

        out = out.transpose(1, 2)

        out, _ = self.lstm(out)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        n_mels,
        prenet_n_layers,
        prenet_embedding_size,
        encoder_embedding_size,
        decoder_n_layers,
        decoder_embedding_size,
        attention_embedding_size,
        attention_location_n_filters,
        attention_location_kernel_size,
        dropout_rate,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_n_layers = decoder_n_layers
        self.decoder_embedding_size = decoder_embedding_size

        self.prenet = Prenet(
            n_mels, prenet_n_layers, prenet_embedding_size, dropout_rate
        )
        self.attention = LocationSensetiveAttention(
            attention_embedding_size,
            decoder_embedding_size,
            encoder_embedding_size,
            attention_location_kernel_size,
            attention_location_n_filters,
        )
        self.rnn_layers = nn.ModuleList()

        self.rnn_layers.append(
            nn.LSTMCell(
                prenet_embedding_size + encoder_embedding_size, decoder_embedding_size
            )
        )

        for _ in range(decoder_n_layers - 1):
            self.rnn_layers.append(
                nn.LSTMCell(decoder_embedding_size, decoder_embedding_size)
            )

        self.proj = nn.Linear(decoder_embedding_size + encoder_embedding_size, n_mels)

        self.gate_layer = nn.Linear(decoder_embedding_size + encoder_embedding_size, 1)

    def get_inital_states(self, memory):
        batch_size = memory.shape[0]
        max_time = memory.shape[1]

        model_device = list(self.parameters())[0].device

        frame = torch.zeros(batch_size, self.n_mels).to(model_device)
        attention_context = torch.zeros(batch_size, self.encoder_embedding_size).to(
            model_device
        )
        alignment_state = torch.zeros(batch_size, max_time).to(model_device)

        rnn_states = []

        for i in range(self.decoder_n_layers):
            rnn_states.append(
                (
                    torch.zeros(batch_size, self.decoder_embedding_size).to(
                        model_device
                    ),
                    torch.zeros(batch_size, self.decoder_embedding_size).to(
                        model_device
                    ),
                )
            )

        return frame, attention_context, rnn_states, alignment_state

    def forward(
        self,
        input_frame,
        encoder_mask,
        attention_context,
        rnn_states,
        alignment_state,
        memory,
        processed_memory,
    ):
        processed_frame = self.prenet(input_frame)
        rnn_hidden_state = torch.cat((processed_frame, attention_context), 1)

        for layer_indx, rnn_layer in enumerate(self.rnn_layers):
            rnn_states[layer_indx] = rnn_layer(rnn_hidden_state, rnn_states[layer_indx])
            rnn_hidden_state = rnn_states[layer_indx][0]

        attention_context, alignment, alignment_state = self.attention(
            rnn_hidden_state, alignment_state, memory, processed_memory, encoder_mask
        )

        decoder_rnn_output = torch.cat((rnn_hidden_state, attention_context), 1)
        output_frame = self.proj(decoder_rnn_output)

        gate_output = self.gate_layer(decoder_rnn_output)
        gate_output = gate_output.squeeze(1)

        return (
            output_frame,
            gate_output,
            alignment,
            attention_context,
            rnn_states,
            alignment_state,
        )
