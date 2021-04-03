import torch
from torch import nn
from ..layers.tacotron2 import Encoder, Decoder, Postnet


class Tacotron2(nn.Module):
    def __init__(self, vocab_size=26, n_mels=80, vocab_embedding_size=512, 
                encoder_embedding_size=512, encoder_n_convolutions=3, encoder_kernel_size=5, 
                prenet_layer_size=256, 
                decoder_embedding_size=512, 
                attention_embedding_size=128, attention_location_n_filters=32, attention_location_kernel_size=31, 
                postnet_n_convolutions=5, postnet_kernel_size=5):
        super().__init__()
        self.encoder = Encoder(vocab_size, vocab_embedding_size, 
                               encoder_embedding_size, encoder_n_convolutions, 
                               encoder_kernel_size)
        self.decoder = Decoder(n_mels, prenet_layer_size, encoder_embedding_size, decoder_embedding_size,
                                attention_embedding_size, attention_location_n_filters, attention_location_kernel_size)

        self.postnet = Postnet(n_mels, decoder_embedding_size, postnet_n_convolutions, postnet_kernel_size)

    def forward(self, texts, encoder_mask, mels_target):
        memory = self.encoder(texts)
        processed_memory = self.decoder.attention.memory_layer(memory)

        frame, attention_context, rnn_state, rnn_memory, alignment_state = self.decoder.get_inital_states(memory)
        frame = frame.unsqueeze(1)
        mels_target = torch.cat((frame, mels_target), 1)[:,:-1]

        decoder_outputs = []
        alignments = []
        gate_outputs = []

        for i in range(mels_target.shape[1]):
            output_frame, gate_output, alignment, attention_context, rnn_state, rnn_memory, alignment_state = self.decoder(mels_target[:,i], 
                                                                                                                           encoder_mask,
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
        decoder_mask = []
        gates = torch.zeros(input_tokens.shape[0], dtype=torch.bool)
        current_step = 0


        while (not gates.all()) and (current_step < max_decoder_steps):
            frame, gate_output, alignment, attention_context, rnn_state, rnn_memory, alignment_state = self.decoder(frame, 
                                                                                                                    mask,
                                                                                                                    attention_context, 
                                                                                                                    rnn_state, 
                                                                                                                    rnn_memory, 
                                                                                                                    alignment_state, 
                                                                                                                    memory, 
                                                                                                                    processed_memory)
            gates = (torch.sigmoid(gate_output) > gate_th) | gates

            alignments.append(alignment)
            decoder_outputs.append(frame)
            decoder_mask.append(gates)
            current_step += 1
        
        decoder_outputs = torch.stack(decoder_outputs, 1)
        decoder_mask = torch.stack(decoder_mask, 1)
        postnet_outputs = self.postnet(decoder_outputs.transpose(1, 2)).transpose(1, 2) + decoder_outputs

        alignments = torch.stack(alignments, 1)
        
        return decoder_outputs, postnet_outputs, alignments, decoder_mask