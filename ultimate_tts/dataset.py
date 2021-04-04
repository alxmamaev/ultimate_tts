import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .utils.audio import normalize_mel


def text_mel_collate_fn(data):
    batch = {}

    texts = [item["text"] for item in data]
    text_lenghts = [len(text) for text in texts]
    batch["texts"] = pad_sequence(texts, batch_first=True)
    batch["encoder_mask"] = torch.zeros_like(batch["texts"]).bool()

    for i, l in enumerate(text_lenghts):
        batch["encoder_mask"][i,l:] = 1

    if data[0].get("mel") is not None:
        mels = [item["mel"] for item in data]
        mel_lenghts = [i.shape[0] for i in mels]
        batch["mels_target"] = pad_sequence(mels, batch_first=True)
        batch["gates_target"] = torch.zeros(batch["mels_target"].shape[0], batch["mels_target"].shape[1])
        batch["decoder_mask"] = torch.zeros_like(batch["mels_target"]).bool()

        for i, l in enumerate(mel_lenghts):
            batch["decoder_mask"][i,l:] = 1
            batch["gates_target"][i][l - 1:] = 1


    if data[0].get("durations") is not None:
        durations = [item["durations"] for item in data]
        batch["durations"] = pad_sequence(durations, batch_first=True)

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

    def __getitem__(self, index):
        filename, text = self.metadata[index]

        mel = np.load(f"{self.mels_datapath}/{filename}.npy")
        tokens = self.text_preprocessor(text)

        tokens = torch.tensor(tokens, dtype=torch.long)
        mel = torch.tensor(mel)
        mel = normalize_mel(mel)

        if self.durations_datapath is not None:
            durations = np.load(f"{self.durations_datapath}/{filename}.npy")
        else:
            durations = None

        return {"text": tokens, "mel": mel, "durations": durations}


    def __len__(self):
        return len(self.metadata)