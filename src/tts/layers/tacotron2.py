import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.dropout import Dropout


class LocalSensetiveAttention(nn.Module):
    def __init__(self, embedding_size, query_size, memory_size, kernel_size, num_filers):
        super().__init__()

        self.query_layer = nn.Linear(query_size, embedding_size)
        self.memory_layer = nn.Linear(memory_size, embedding_size)
        self.location_layer_conv = nn.Conv1d(1, num_filers, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.location_layer_linear = nn.Linear(num_filers, embedding_size, bias=False)

        self.v = nn.Linear(embedding_size, 1, bias=False)

    def forward(self, query, alignment_state, memory, processed_memory, mask):
        processed_query = self.query_layer(query)
        processed_query = processed_query.unsqueeze(1)

        expanded_alignments = alignment_state.unsqueeze(1)

        processed_location_features = self.location_layer_conv(expanded_alignments)
        processed_location_features = processed_location_features.transpose(1, 2)
        processed_location_features = self.location_layer_linear(processed_location_features)

        energy = self.v(torch.tanh(processed_location_features + processed_query + processed_memory))
        energy = energy.squeeze(2)        
        
        energy.data.masked_fill_(mask, float("-inf"))
        alignments = torch.softmax(energy, 1)
        
        next_state = alignment_state + alignments

        attention_context = torch.bmm(alignments.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, alignments, next_state


class Postnet(nn.Module):
    def __init__(self, num_mels, embedding_size, n_convolutions, kernel_size):
        super().__init__()

        self.convolutions = nn.ModuleList()

        self.convolutions.append(nn.Sequential(
            nn.Conv1d(num_mels, embedding_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        ))

        for _ in range(n_convolutions - 2):
            self.convolutions.append(nn.Sequential(
                nn.Conv1d(embedding_size, embedding_size, kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(embedding_size),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            ))

        self.convolutions.append(nn.Sequential(
            nn.Conv1d(embedding_size, num_mels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(num_mels),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        ))


    def forward(self, X):
        out = X
        for conv in self.convolutions:
            out = conv(out)

        return out

    def inference(self, X):
        pass


class Prenet(nn.Module):
    def __init__(self, num_mels, layers_sizes):
        super().__init__()

        input_sizes = [num_mels] + layers_sizes[:-1]
        output_sizes = layers_sizes

        self.layers = nn.ModuleList()

        for input_size, output_size in zip(input_sizes, output_sizes):
            layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU()
            )
            self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
            # We are using dropout at the test time 
            out = torch.dropout(out, p=0.5, train=True)

        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, vocab_embedding_size, 
                encoder_embedding_size, n_convolutions, 
                kernel_size):
        assert encoder_embedding_size % 2 == 0, "encoder_embedding_size must be divisible by 2"
        
        super().__init__()

        self.embdedding = nn.Embedding(vocab_size, vocab_embedding_size)
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(vocab_embedding_size, 
                          encoder_embedding_size,
                          kernel_size,
                          padding = (kernel_size - 1) // 2),
                
                nn.BatchNorm1d(encoder_embedding_size),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        )

        for _ in range(n_convolutions -1):
            conv_layer = nn.Sequential(
                nn.Conv1d(encoder_embedding_size, 
                          encoder_embedding_size,
                          kernel_size,
                          padding = (kernel_size - 1) // 2),
                
                nn.BatchNorm1d(encoder_embedding_size),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
            self.convolutions.append(conv_layer)
        
        self.lstm = nn.LSTM(encoder_embedding_size, 
                            encoder_embedding_size // 2, 
                            bidirectional=True,
                            batch_first=True)


    def forward(self, X):
        out = self.embdedding(X)
        out = out.transpose(2, 1)
        
        for conv in self.convolutions:
            out = conv(out)

        out = out.transpose(1, 2)

        out, _ = self.lstm(out)

        return out

    def inference(self, X):
        pass


class Decoder(nn.Module):
    def __init__(self, num_mels, prenet_layer_size, 
                encoder_embedding_size, decoder_embedding_size, 
                attention_embedding_size, attention_location_n_filters, 
                attention_location_kernel_size):
        super().__init__()

        self.num_mels = num_mels
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_embedding_size = decoder_embedding_size

        self.prenet = Prenet(num_mels=num_mels, layers_sizes=[prenet_layer_size, prenet_layer_size])
        self.rnn = nn.LSTMCell(prenet_layer_size + encoder_embedding_size, 
                              decoder_embedding_size)
        self.attention = LocalSensetiveAttention(attention_embedding_size, decoder_embedding_size, encoder_embedding_size, attention_location_kernel_size, attention_location_n_filters)
        self.proj = nn.Linear(decoder_embedding_size + encoder_embedding_size, 
                              num_mels)

        self.gate_layer = nn.Linear(decoder_embedding_size + encoder_embedding_size, 1)

    def get_inital_states(self, memory):
        batch_size = memory.shape[0]
        max_time = memory.shape[1]

        model_device = list(self.parameters())[0].device

        frame = torch.zeros(batch_size, self.num_mels).to(model_device)
        attention_context = torch.zeros(batch_size, self.encoder_embedding_size).to(model_device)
        rnn_state = torch.zeros(batch_size, self.decoder_embedding_size).to(model_device)
        rnn_memory = torch.zeros(batch_size, self.decoder_embedding_size).to(model_device)
        alignment_state = torch.zeros(batch_size, max_time).to(model_device)
        
        return frame, attention_context, rnn_state, rnn_memory, alignment_state

    def forward(self, input_frame, encoder_mask, attention_context, rnn_state, rnn_memory, alignment_state, memory, processed_memory):
        processed_frame = self.prenet(input_frame)
        rnn_input = torch.cat((processed_frame, attention_context), 1)

        rnn_state, rnn_memory = self.rnn(rnn_input, (rnn_state, rnn_memory))
        attention_context, alignment, alignment_state = self.attention(rnn_state, alignment_state, memory, processed_memory, encoder_mask)

        decoder_rnn_output = torch.cat((rnn_state, attention_context), 1)
        output_frame = self.proj(decoder_rnn_output)

        gate_output = self.gate_layer(decoder_rnn_output)

        return output_frame, gate_output, alignment, attention_context, rnn_state, rnn_memory, alignment_state
