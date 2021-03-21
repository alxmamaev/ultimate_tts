import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.dropout import Dropout
from torch.nn.parameter import Parameter

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
        
        energy[mask] = float("-inf")
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
            # We are using dropout at thetest time 
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
                            bidirectional=True)


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

    def forward(self, input_frame, mask, attention_context, rnn_state, rnn_memory, alignment_state, memory, processed_memory):
        processed_frame = self.prenet(input_frame)
        rnn_input = torch.cat((processed_frame, attention_context), 1)

        rnn_state, rnn_memory = self.rnn(rnn_input, (rnn_state, rnn_memory))
        attention_context, alignment, alignment_state = self.attention(rnn_state, alignment_state, memory, processed_memory, mask)

        decoder_rnn_output = torch.cat((rnn_state, attention_context), 1)
        output_frame = self.proj(decoder_rnn_output)

        gate_output = self.gate_layer(decoder_rnn_output)

        return output_frame, gate_output, alignment, attention_context, rnn_state, rnn_memory, alignment_state


class Tacotron2(nn.Module):
    def __init__(self, vocab_size=26, num_mels=80, vocab_embedding_size=512, 
                encoder_embedding_size=512, encoder_n_convolutions=3, encoder_kernel_size=5, 
                prenet_layer_size=256, 
                decoder_embedding_size=512, 
                attention_embedding_size=128, attention_location_n_filters=32, attention_location_kernel_size=31, 
                postnet_n_convolutions=5, postnet_kernel_size=5):
        super().__init__()
        self.encoder = Encoder(vocab_size, vocab_embedding_size, 
                               encoder_embedding_size, encoder_n_convolutions, 
                               encoder_kernel_size)
        self.decoder = Decoder(num_mels, prenet_layer_size, encoder_embedding_size, decoder_embedding_size,
                                attention_embedding_size, attention_location_n_filters, attention_location_kernel_size)

        self.postnet = Postnet(num_mels, decoder_embedding_size, postnet_n_convolutions, postnet_kernel_size)

    def forward(self, texts, mask, mels):
        memory = self.encoder(texts)
        processed_memory = self.decoder.attention.memory_layer(memory)

        frame, attention_context, rnn_state, rnn_memory, alignment_state = self.decoder.get_inital_states(memory)
        frame = frame.unsqueeze(1)
        mels = torch.cat((frame, mels), 1)[:,:-1]

        decoder_outputs = []
        alignments = []
        gate_outputs = []

        for i in range(mels.shape[1]):
            output_frame, gate_output, alignment, attention_context, rnn_state, rnn_memory, alignment_state = self.decoder(mels[:,i], 
                                                                                                                           mask,
                                                                                                                           attention_context, 
                                                                                                                           rnn_state, 
                                                                                                                           rnn_memory, 
                                                                                                                           alignment_state, 
                                                                                                                           memory, 
                                                                                                                           processed_memory)
            alignments.append(alignment)
            
            decoder_outputs.append(output_frame)
            gate_outputs.append(gate_output)

        decoder_outputs = torch.stack(decoder_outputs, 1)
        
        postnet_outputs = self.postnet(decoder_outputs.transpose(1, 2)).transpose(1, 2) + decoder_outputs

        alignments = torch.stack(alignments, 1)
        gate_outputs = torch.stack(gate_outputs, 1)

        return decoder_outputs, postnet_outputs, alignments, gate_outputs

    @torch.no_grad()
    def inference(self, input_tokens, mask, gate_th=0.5, max_decoder_steps=1000):
        memory = self.encoder(input_tokens)
        processed_memory = self.decoder.attention.memory_layer(memory)

        frame, attention_context, rnn_state, rnn_memory, alignment_state = self.decoder.get_inital_states(memory)

        decoder_outputs = []
        alignments = []
        gate_output = torch.ones(input_tokens.shape[0], 1) * -1
        current_step = 0


        while (torch.sigmoid(gate_output) < gate_th).any() and current_step < max_decoder_steps:
            frame, gate_output, alignment, attention_context, rnn_state, rnn_memory, alignment_state = self.decoder(frame, 
                                                                                                                    mask,
                                                                                                                    attention_context, 
                                                                                                                    rnn_state, 
                                                                                                                    rnn_memory, 
                                                                                                                    alignment_state, 
                                                                                                                    memory, 
                                                                                                                    processed_memory)
            alignments.append(alignment)
            decoder_outputs.append(frame)
            current_step += 1
        
        decoder_outputs = torch.stack(decoder_outputs, 1)
        postnet_outputs = self.postnet(decoder_outputs.transpose(1, 2)).transpose(1, 2) + decoder_outputs

        alignments = torch.stack(alignments, 1)

        output = {
            "decoder_outputs": decoder_outputs,
            "postnet_outputs": postnet_outputs,
            "alignments": alignments
        }

        return output