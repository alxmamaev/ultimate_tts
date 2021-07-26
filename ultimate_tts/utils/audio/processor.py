import torchaudio as taudio
from catalyst.registry import REGISTRY
from pathlib import Path
from tqdm import tqdm
from math import ceil
import os
from functools import partial
from torch.multiprocessing import Pool

class AudioProcessor:
    def __init__(self, transforms, batch_size=1):
        """AudioProcessor handles audios processing.
        This processor applies transforms in their order to the batch of audios.

        Parameters
        ----------
        transforms : Union[List[Callable], List[Dict]]
            List of callable transforms objects, or their config dicts.
        batch_size : int, optional
            Batch size for data processing, by default 1
        """

        self.transforms = []
        self.batch_size = batch_size

        for transform in transforms:
            if isinstance(transform, dict):
                transform = REGISTRY.get_from_params(**transform)

            self.transforms.append(transform)


    def __call__(self, audios_batch):
        """Function applies transforms to all input audios in the batch, and returns
           list of processed audios

        Parameters
        ----------
        audios_batch : List[torch.FloatTensor]
            Input batch of audios strings

        Returns
        -------
        List[torch.FloatTensor]
            Batch with processed audios
        """
        for transform in self.transforms:
            audios_batch = transform(audios_batch)

        return audios_batch

    
    def process_batch_offline(self, sample_rate, output_path, input_paths):
        audios_batch = []
        for input_wav_path in input_paths:
            audio, _sample_rate = taudio.load(str(input_wav_path))

            assert sample_rate <= _sample_rate, f"Error at file {input_wav_path}, sample rate of input audio less than target sample_rate"

            audio = taudio.functional.resample(audio, _sample_rate, sample_rate)

            assert audio.shape[0] == 1, f"Error at file {input_wav_path}, audio must be monophonic"
            assert audio.shape[1] > 0, f"Error at file {input_wav_path}, audio length is zero"

            audio = audio[0] # Get first channel of audio
            audios_batch.append(audio)

        audios_batch = self.__call__(audios_batch)
        
        for input_path, audio in  zip(input_paths, audios_batch):
            filename = input_path.stem + ".wav"
            output_wav_path = output_path.joinpath(filename)

            audio = audio.unsqueeze(0)
            assert audio.shape[1] != 0, f"Error at file {input_wav_path}, audio length after processing is zero"
            taudio.save(str(output_wav_path), audio, sample_rate)


    def process_files(self, inputs, outputs, sample_rate=22050, verbose=False):
        input_metadata = []

        input_metadata_path = Path(inputs["metadata_path"])
        input_wavs_path = Path(inputs["wavs_path"])
        output_wavs_path = Path(outputs["wavs_path"])

        output_wavs_path.mkdir(parents=True, exist_ok=True)

        with open(input_metadata_path) as f:
            for line in f:
                line = line.strip()
                filename, text = line.split("|", 1)
                input_metadata.append((filename, text))
        

        item_index = 0
        batched_input_paths = []

        while True:
            input_paths_batch = []

            while item_index < len(input_metadata) and len(input_paths_batch) < self.batch_size:
                filename, _ = input_metadata[item_index]
                input_wav_path = input_wavs_path.joinpath(filename + ".wav")
                
                input_paths_batch.append(input_wav_path)
                item_index += 1
            
            if not input_paths_batch:
                break

            batched_input_paths.append(input_paths_batch)
        

        process_func = partial(self.process_batch_offline, sample_rate, output_wavs_path)
        nproc = os.cpu_count()

        with Pool(nproc) as p:
            tasks = p.imap_unordered(process_func, batched_input_paths)

            if verbose:
                print("Start audio processing")
                tasks = tqdm(tasks, total=len(batched_input_paths))

            for _ in tasks:
                continue