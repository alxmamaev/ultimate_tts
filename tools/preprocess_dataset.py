import yaml
from glob import glob
import torch
import torchaudio as taudio
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np


def parse():
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("metadata")
    parser.add_argument("datapath")
    parser.add_argument("--valid_size", type=int, default=100)

    return parser.parse_args()


def main(args):
    with open(args.config, encoding="utf-8") as f:
        config = yaml.load(f, yaml.Loader)

    feature_extractor_params = config["feature_extractor_params"]

    data_params = config["data_params"]
    mel_extractor = taudio.transforms.MelSpectrogram(sample_rate=feature_extractor_params["sample_rate"],
                                                     n_fft=feature_extractor_params["n_fft"],
                                                     n_mels=feature_extractor_params["n_mels"],
                                                     win_length=feature_extractor_params["win_length"],
                                                     hop_length=feature_extractor_params["hop_length"],
                                                     f_min=feature_extractor_params["f_min"],
                                                     f_max=feature_extractor_params["f_max"], 
                                                     power=feature_extractor_params["power"])
    

    trim_front_params = feature_extractor_params.get("trim_front_params", None)
    trim_back_params = feature_extractor_params.get("trim_back_params", None)

    vad_front = None if trim_front_params is None else taudio.transforms.Vad(sample_rate=feature_extractor_params["sample_rate"], **trim_front_params)
    vad_back = None if trim_back_params is None else taudio.transforms.Vad(sample_rate=feature_extractor_params["sample_rate"], **trim_back_params)

    with open(args.metadata, encoding="utf-8") as f:
        lines = f.readlines()

    input_datapath = Path(args.datapath)
    train_metadata_path = Path(data_params["train_metadata"])
    valid_metadata_path = Path(data_params["valid_metadata"])
    mels_datapath = Path(data_params["mels_datapath"])
    wavs_datapath = Path(data_params["wavs_datapath"]) if data_params.get("wavs_datapath", False) else None

    with open(train_metadata_path, "w", encoding="utf-8") as f:
        f.writelines(lines[args.valid_size:])

    with open(valid_metadata_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:args.valid_size])
        
    for line in tqdm(lines):
        file_name, text = line.strip().split("|")[:2]
        input_wav_path = input_datapath.joinpath(file_name + ".wav")
        mel_path = mels_datapath.joinpath(file_name + ".npy")
        wav_path = None if wavs_datapath is None else wavs_datapath.joinpath(file_name + ".wav")

        mel_path.parent.mkdir(exist_ok=True)
        wav_path.parent.mkdir(exist_ok=True)

        audio, sample_rate = taudio.load_wav(str(input_wav_path))

        assert sample_rate == feature_extractor_params["sample_rate"], "Expected sample_rate {}, but gets {}".format(feature_extractor_params["sample_rate"], sample_rate)
        assert len(audio[0].shape) == 1, "Audio must be monophonic"

        audio = audio[0] # Getting first channel

        # Trimming
        if vad_front is not None:
            audio = vad_front(audio)

        if vad_back is not None:
            audio = torch.flip(audio, [0])
            audio = vad_back(audio)
            audio = torch.flip(audio, [0])

        audio = audio / feature_extractor_params["wav_max_value"] # Normalize audio
        mel = mel_extractor(audio).transpose(0, 1).numpy()

        np.save(str(mel_path), mel)
        if wav_path is not None:
            taudio.save(str(wav_path), audio.unsqueeze(0), sample_rate)
        

if __name__ == "__main__":
    args = parse()
    main(args)