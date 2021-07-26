import torch


def normalize_mel(mel):
    return torch.log10(mel + 1e-3)

def denormalize_mel(mel):
    return torch.pow(10.0, mel) - 1e-3