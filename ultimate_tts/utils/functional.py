import torch

def pad():
    pass
    #return padded_sequence, sequence_lenghts

def lenght_to_mask():
    pass

def mask_to_lenghts():
    pass

def normalize_mel(mel):
    return torch.log10(mel + 1e-3)

def denormalize_mel(mel):
    return torch.pow(10.0, mel) - 1e-3