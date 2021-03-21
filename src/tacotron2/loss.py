from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mel_loss = nn.MSELoss()
        self.gate_loss = nn.BCEWithLogitsLoss()

    def forward(self, model_output, target):
        decoder_outputs, postnet_outputs, alignments, gate_outputs = model_output
        mel_target, gate_target = target

        gate_outputs = gate_outputs.view(-1, 1)
        gate_target = gate_target.view(-1, 1)

        mel_target.requires_grad = False
        gate_target.requires_grad = False

        mel_loss = self.mel_loss(decoder_outputs, mel_target) + self.mel_loss(postnet_outputs, mel_target)
        gate_loss = self.gate_loss(gate_outputs, gate_target)

        return mel_loss + gate_loss

        