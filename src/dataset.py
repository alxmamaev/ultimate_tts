import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def text_mel_collate_fn(data):
    texts = [i[0] for i in data]
    text_lenghts = [len(text) for text in texts]

    mels = [i[1] for i in data]

    padded_texts = pad_sequence(texts, batch_first=True)
    mel_lenghts = [i.shape[0] for i in mels]
    max_mel_length = max(mel_lenghts)
    
    padded_mels = torch.zeros((len(mels), max_mel_length, 80))
    gates_target = torch.zeros((len(mels), max_mel_length))

    for i, mel in enumerate(mels):
        padded_mels[i][:mel.shape[0]] = mel
        gates_target[i][mel.shape[0] - 1:] = 1

    encoder_mask = torch.zeros_like(padded_texts).bool()
    for i, l in enumerate(text_lenghts):
        encoder_mask[i,l:] = 1

    decoder_mask = torch.zeros_like(padded_mels).bool()
    for i, l in enumerate(mel_lenghts):
        decoder_mask[i,l:] = 1


    batch = {
        "texts": padded_texts,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "mels": padded_mels,
        "text_lenghts": text_lenghts,
        "mel_lenghts": mel_lenghts,
        "targets": [padded_mels, gates_target]
    }

    return batch


class TextMelDataset(Dataset):
    def __init__(self, metadata_path, datapath, text_preprocessor):
        self.metadata = []

        with open(metadata_path) as f:
            for line in f:
                filename, text = line.strip().split("|")[:2]
                self.metadata.append((filename, text))
        
        self.datapath = datapath
        self.text_preprocessor = text_preprocessor

    def normalize_mel(self, mel):
        return torch.log10(mel)

    def __getitem__(self, index):
        filename, text = self.metadata[index]

        mel = np.load(f"{self.datapath}/{filename}.npy")
        tokens = self.text_preprocessor(text)

        tokens = torch.tensor(tokens, dtype=torch.long)
        mel = torch.tensor(mel)
        mel = self.normalize_mel(mel)

        return tokens, mel


    def __len__(self):
        return len(self.metadata)