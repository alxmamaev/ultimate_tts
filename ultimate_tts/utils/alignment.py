import torch

def alignment_to_durations(alignment):
    durations = torch.zeros(alignment.shape[0], dtype=torch.long)
    current_token_index = 0
    
    hard_attention = torch.argmax(alignment, 0)
    
    for token_index in hard_attention.tolist():          
        if token_index > current_token_index:
            current_token_index = token_index
            
        durations[current_token_index] += 1
    
    return durations