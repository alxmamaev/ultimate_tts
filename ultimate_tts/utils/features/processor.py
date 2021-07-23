from pathlib import Path
import numpy as np
from tqdm import tqdm
import torchaudio as taudio
from catalyst.registry import REGISTRY
from math import ceil


class FeaturesProcessor:
    def __init__(self, batch_size=1, mel_extractor=None, speaker_embedding_extractor=None, prossody_extractor=None, wav_max_value=32768):
        self.batch_size = batch_size
        self.wav_max_value = wav_max_value

        if isinstance(mel_extractor, dict):
            mel_extractor = REGISTRY.get_from_params(**mel_extractor)
        self.mel_extractor = mel_extractor

        if isinstance(speaker_embedding_extractor, dict):
            speaker_embedding_extractor = REGISTRY.get_from_params(**speaker_embedding_extractor)
        self.speaker_embedding_extractor = speaker_embedding_extractor

        if isinstance(prossody_extractor, dict):
            prossody_extractor = REGISTRY.get_from_params(**prossody_extractor)
        self.prossody_extractor = prossody_extractor
        

    def __call__(self, audios_batch, durations_batch=[]):
        features = {}

        if self.mel_extractor is not None:
            features["mels"] = self.mel_extractor(audios_batch)

        if self.speaker_embedding_extractor is not None:
            features["speaker_embeddings"] = self.speaker_embedding_extractor(audios_batch)

        if self.prossody_extractor is not None:
            assert durations_batch != [] and durations_batch is not None, "expected durations for prosody extraction, but durations size is zero"
            assert len(audios_batch) == len(durations_batch), "audios and durations batch size must be same"
            assert all([audio.shape[0] == dur.sum() for audio, dur in zip(audios_batch, durations_batch)]), "sum of duration must be equal audio size"

            features["prossodies"] = self.prossody_extractor(audios_batch, durations_batch)

        return features


    def process_files(self, inputs, outputs, verbose=False):
        input_metadata = []

        input_metadata_path = Path(inputs["metadata_path"])
        input_wavs_path = Path(inputs["wavs_path"])
        input_durations_path = Path(inputs["durations_path"]) if inputs.get("durations_path", False) else None

        output_mels_datapath = Path(outputs["mels_path"]) if outputs.get("mels_path", False) else None
        output_speakers_embeddings_path = Path(outputs["speakers_embeddings_path"]) if outputs.get("speakers_embeddings_path", False) else None
        output_prossodies_datapath = Path(outputs["prosodies_path"]) if outputs.get("prosodies_path", False) else None

        for i in [output_mels_datapath, output_speakers_embeddings_path, output_prossodies_datapath]:
            i.mkdir(parents=True, exist_ok=True)


        with open(input_metadata_path) as f:
            for line in f:
                line = line.strip()
                input_metadata.append(tuple(line.split("|", 1)))
                

        item_index = 0
        if verbose:
            print("Processing features...")
            progress_bar = tqdm(total=int(ceil(len(input_metadata) / self.batch_size)))

        while True:
            audios_batch = []
            durations_batch = []
            filenames_batch = []

            while item_index < len(input_metadata) and len(filenames_batch) < self.batch_size:
                filename, _ = input_metadata[item_index]

                input_wav_path = input_wavs_path.joinpath(filename + ".wav")
                durations_path = input_durations_path.joinpath(filename + ".npy") if input_durations_path is not None else None

                audio, _ = taudio.load(str(input_wav_path))

                assert audio.shape[0] == 1, "audio must be monophonic"
                audio = audio[0] 
                audio = audio.float() / self.wav_max_value

                durations = np.load(str(durations_path)) if durations_path is not None else None
                
                audios_batch.append(audio)
                durations_batch.append(durations)
                filenames_batch.append(filename)
                item_index += 1
            
            if not filenames_batch:
                break

            features = self.__call__(audios_batch, durations_batch)

            for i, filename in enumerate(filenames_batch):
                if features.get("mels") is not None:
                    output_mel_path = output_mels_datapath.joinpath(filename + ".npy")
                    np.save(str(output_mel_path), features["mels"][i])

                if features.get("speaker_embeddings") is not None:
                    output_xvector_path = output_speakers_embeddings_path.joinpath(filename + ".npy")
                    np.save(str(output_xvector_path), features["xvector"][i])
                
                if features.get("prossody") is not None:
                    output_prosody_path = output_prossodies_datapath.joinpath(filename + ".npy")
                    np.save(str(output_prosody_path), features["prossody"][i])

            if verbose:
                progress_bar.update(1)


__all__ = ["FeaturesProcessor"]
