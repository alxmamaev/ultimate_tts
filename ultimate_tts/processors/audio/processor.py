import logging
import os
from functools import partial
from math import ceil
from pathlib import Path

import torchaudio as taudio
from catalyst.registry import REGISTRY
from torch.multiprocessing import get_context
from tqdm import tqdm

from ...data.corpus import Corpus
from ...utils.functional import split_to_batches


class AudioProcessor:
    def __init__(self, transforms, sample_rate=22050, batch_size=1):
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
        self.sample_rate = sample_rate

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

            assert (
                sample_rate <= _sample_rate
            ), f"Error at file {input_wav_path}, sample rate of input audio less than target sample_rate"

            audio = taudio.transforms.Resample(_sample_rate, sample_rate)(audio)

            assert (
                audio.shape[0] == 1
            ), f"Error at file {input_wav_path}, audio must be monophonic"
            assert (
                audio.shape[1] > 0
            ), f"Error at file {input_wav_path}, audio length is zero"

            audio = audio[0]  # Get first channel of audio
            audios_batch.append(audio)

        audios_batch = self.__call__(audios_batch)

        for input_path, audio in zip(input_paths, audios_batch):
            filename = input_path.stem + ".wav"
            output_wav_path = output_path.joinpath(filename)

            audio = audio.unsqueeze(0)
            assert (
                audio.shape[1] != 0
            ), f"Error at file {input_wav_path}, audio length after processing is zero"
            taudio.save(str(output_wav_path), audio, sample_rate)

    def process_files(self, inputs, outputs, verbose=False):
        logging.info("Start audio processor")

        input_corpus_path = Path(inputs["corpus_path"])
        input_wavs_path = Path(inputs["wavs_path"])
        output_wavs_path = Path(outputs["wavs_path"])

        output_wavs_path.mkdir(parents=True, exist_ok=True)

        logging.info("Read corpus file")
        input_corpus = Corpus.from_file(input_corpus_path)

        input_paths_batches = []

        for transcriptions_batch in split_to_batches(
            input_corpus, batch_size=self.batch_size
        ):
            input_paths_batch = []
            for transcription in transcriptions_batch:
                filename = transcription.filename
                wav_path = input_wavs_path.joinpath(filename + ".wav")
                input_paths_batch.append(wav_path)
            input_paths_batches.append(input_paths_batch)

        del input_corpus

        process_func = partial(
            self.process_batch_offline, self.sample_rate, output_wavs_path
        )
        nproc = os.cpu_count()

        logging.info(f"Start audio processing in {nproc} jobs")
        with get_context("spawn").Pool(nproc) as pool:
            tasks = pool.imap_unordered(process_func, input_paths_batches)

            if verbose:
                tasks = tqdm(tasks, total=len(input_paths_batches))

            for _ in tasks:
                continue
