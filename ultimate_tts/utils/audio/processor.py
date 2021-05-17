import torchaudio as taudio
from catalyst.registry import REGISTRY
from pathlib import Path
from tqdm import tqdm
from math import ceil


class AudioProcessor:
    def __init__(self, sample_rate, transforms, batch_size=1):
        self.sample_rate = sample_rate
        self.transforms = []
        self.batch_size = batch_size

        for transform in transforms:
            if isinstance(transform, dict):
                transform = REGISTRY.get_from_params(**transform)

            self.transforms.append(transform)


    def __call__(self, audios_batch):
        for transform in self.transforms:
            audios_batch = transform(audios_batch)

        return audios_batch


    def process_files(self, inputs, outputs, verbose=False):
        input_metadata = []
        filtered_files = set()

        input_metadata_path = Path(inputs["metadata_path"])
        input_wavs_path = Path(inputs["wavs_path"])
        output_wavs_path = Path(outputs["wavs_path"])
        output_metadata_path = Path(outputs["metadata_path"])

        output_wavs_path.mkdir(parents=True, exist_ok=True)
        output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_metadata_path) as f:
            for line in f:
                line = line.strip()
                filename, text = line.split("|", 1)
                input_metadata.append((filename, text))
        

        item_index = 0
        if verbose:
            print("Processing audios...")
            progress_bar = tqdm(total=int(ceil(len(input_metadata) / self.batch_size)))

        while True:
            audios_batch = []
            filenames_batch = []

            while item_index < len(input_metadata) and len(filenames_batch) < self.batch_size:
                filename, _ = input_metadata[item_index]

                input_wav_path = input_wavs_path.joinpath(filename + ".wav")
                audio, _ = taudio.load_wav(str(input_wav_path))

                assert audio.shape[0] == 1, "audio must be monophonic"
                audio = audio[0]

                audios_batch.append(audio)
                filenames_batch.append(filename)
                item_index += 1
            
            if not filenames_batch:
                break

            audios_batch = self.__call__(audios_batch)
            for filename, audio in  zip(filenames_batch, audios_batch):
                if audio is None:
                    continue

                output_wav_path = output_wavs_path.joinpath(filename + ".wav")
                taudio.save(str(output_wav_path), audio.unsqueeze(0), self.sample_rate)
                filtered_files.add(filename)

            if verbose:
                progress_bar.update(1)


        with open(output_metadata_path, "w") as f:
            for filename, text in input_metadata:
                if filename in filtered_files:
                    f.write(f"{filename}|{text}\n")