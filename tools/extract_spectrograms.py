import yaml
from glob import glob
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
    parser.add_argument("--output_train_metadata", default="./data/train_metadata.csv")
    parser.add_argument("--output_valid_metadata", default="./data/valid_metadata.csv")
    parser.add_argument("--output_datapath", default="./data/mels")
    parser.add_argument("--valid_size", type=int, default=100)

    return parser.parse_args()


def main(args):
    with open(args.config) as f:
        config = yaml.load(f, yaml.Loader)

    feature_extractor_params = config["feature_extractor_params"]
    mel_extractor = taudio.transforms.MelSpectrogram(sample_rate=feature_extractor_params["sample_rate"],
                                                     n_fft=feature_extractor_params["n_fft"],
                                                     n_mels=feature_extractor_params["n_mels"],
                                                     win_length=feature_extractor_params["win_length"],
                                                     hop_length=feature_extractor_params["hop_length"],
                                                     f_min=feature_extractor_params["f_min"],
                                                     f_max=feature_extractor_params["f_max"], 
                                                     power=feature_extractor_params["power"])
    
    with open(args.metadata) as f:
        lines = f.readlines()


    input_datapath = Path(args.datapath)
    output_datapath = Path(args.output_datapath)

    with open(args.output_train_metadata, "w") as f:
        f.writelines(lines[args.valid_size:])

    with open(args.output_valid_metadata, "w") as f:
        f.writelines(lines[:args.valid_size])
        
    for line in tqdm(lines):
        file_name, text = line.strip().split("|")[:2]
        wav_path = input_datapath.joinpath(file_name + ".wav")
        mel_path = output_datapath.joinpath(file_name + ".npy")

        mel_path.parent.mkdir(exist_ok=True)

        audio, sample_rate = taudio.load_wav(str(wav_path))
        audio = audio / feature_extractor_params["wav_max_value"]
        mel = mel_extractor(audio)[0].transpose(0, 1).numpy()

        np.save(str(mel_path), mel)
            

if __name__ == "__main__":
    args = parse()
    main(args)