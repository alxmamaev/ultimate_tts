import torch

def normalize_mel(mel):
    return torch.log10(mel)

def denormalize_mel(mel):
    return torch.pow(10.0, mel)