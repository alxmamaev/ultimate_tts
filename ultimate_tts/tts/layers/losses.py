from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mel_loss = nn.MSELoss()
        self.gate_loss = nn.BCEWithLogitsLoss()

    def forward(self, decoder_outputs, postnet_outputs, alignments, gate_outputs, mels_target, gates_target, decoder_mask):
        decoder_outputs.data.masked_fill_(decoder_mask, 0.0)
        postnet_outputs.data.masked_fill_(decoder_mask, 0.0)
        gate_outputs.data.masked_fill_(decoder_mask[:, :, 0], 1e3)

        mels_target.requires_grad = False
        gates_target.requires_grad = False

        mel_loss = self.mel_loss(decoder_outputs, mels_target) + self.mel_loss(postnet_outputs, mels_target)
        gate_loss = self.gate_loss(gate_outputs, gates_target)

        return mel_loss + gate_loss

        

class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, model_output, target):
        output_mels, durations, alignments = model_output
        mel_target, target_durations, encoder_mask, decoder_mask = target

        output_mels.data.masked_fill_(decoder_mask, 0.0)
        target_durations.data.masked_fill_(encoder_mask[:, :, 0], 0)

        mel_target.requires_grad = False
        target_durations.requires_grad = False

        mel_loss = self.mse(output_mels, mel_target) 
        duration_loss = self.mse(durations, target_durations)

        return mel_loss + duration_loss


__all__ = ["Tacotron2Loss", "FastSpeechLoss"]