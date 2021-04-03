import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def text_mel_collate_fn(data):
    texts = [item["text"] for item in data]
    mels = [item["mel"] for item in data]

    if data[0].get("durations") is None:
        durations = None
        padded_durations = None
    else:
        durations = [item["durations"] for item in data]
        padded_durations = pad_sequence(durations, batch_first=True)

    text_lenghts = [len(text) for text in texts]
    mel_lenghts = [i.shape[0] for i in mels]

    padded_texts = pad_sequence(texts, batch_first=True)
    padded_mels = pad_sequence(mels, batch_first=True)
    gates = torch.zeros(padded_mels.shape[0], padded_mels.shape[1])
    encoder_mask = torch.zeros_like(padded_texts).bool()
    decoder_mask = torch.zeros_like(padded_mels).bool()

    for i, l in enumerate(text_lenghts):
        encoder_mask[i,l:] = 1
    
    for i, l in enumerate(mel_lenghts):
        decoder_mask[i,l:] = 1
        gates[i][l - 1:] = 1


    batch = {
        "texts": padded_texts,
        "mels_target": padded_mels,
        "gates_target": gates,
        "durations": padded_durations,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask
    }

    return batch


class TextMelDataset(Dataset):
    def __init__(self, text_preprocessor, metadata_path, mels_datapath, durations_datapath=None):
        self.metadata = []

        with open(metadata_path) as f:
            for line in f:
                filename, text = line.strip().split("|")[:2]
                self.metadata.append((filename, text))
        
        self.mels_datapath = mels_datapath
        self.durations_datapath = durations_datapath

        self.text_preprocessor = text_preprocessor

    def normalize_mel(self, mel):
        return torch.log10(mel)

    def __getitem__(self, index):
        filename, text = self.metadata[index]

        mel = np.load(f"{self.mels_datapath}/{filename}.npy")
        tokens = self.text_preprocessor(text)

        tokens = torch.tensor(tokens, dtype=torch.long)
        mel = torch.tensor(mel)
        mel = self.normalize_mel(mel)

        if self.durations_datapath is not None:
            durations = np.load(f"{self.durations_datapath}/{filename}.npy")
        else:
            durations = None

        return {"text": tokens, "mel": mel, "durations": durations}


    def __len__(self):
        return len(self.metadata)