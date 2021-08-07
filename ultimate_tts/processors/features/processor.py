import logging
from math import ceil
from pathlib import Path

import numpy as np
import torchaudio as taudio
from catalyst.registry import REGISTRY
from tqdm import tqdm

from ...data.corpus import Corpus
from ...utils.functional import split_to_batches


class FeaturesProcessor:
    def __init__(self, transforms, batch_size: int = 1):
        self.transforms = []
        self.batch_size = batch_size

        for transform in transforms:
            if isinstance(transform, dict):
                transform = REGISTRY.get_from_params(**transform)

            self.transforms.append(transform)

    def __call__(self, corpus, audios_batch, durations_batch=[]):
        features = {}

        assert len(corpus) == len(
            audios_batch
        ), "Audios count must be same as corpus size."

        for transform in self.transforms:
            extrator_output = transform(corpus, audios_batch)
            features.update(extrator_output)

        return features

    def process_files(self, inputs, outputs, verbose=True):
        logging.info("Start feature processor")

        input_corpus_path = Path(inputs["corpus_path"])
        input_wavs_path = Path(inputs["wavs_path"])
        output_features_path = Path(outputs["features_path"])

        output_features_path.mkdir(parents=True, exist_ok=True)


        logging.info("Read corpus file")
        input_corpus = Corpus.from_file(input_corpus_path)


        logging.info("Start feature extracting")

        total_batches = ceil(len(input_corpus) / self.batch_size)

        for transcriptions_batch in tqdm(split_to_batches(
            input_corpus, batch_size=self.batch_size
        ), total=total_batches):
            audios_batch = []
            for transcription in transcriptions_batch:
                filename = transcription.filename
                wav_path = input_wavs_path.joinpath(filename + ".wav")
                wav, rate = taudio.load(str(wav_path))
                wav = wav[0]

                audios_batch.append(wav)

            output_features = self(transcriptions_batch, audios_batch)

            for feature_name in output_features.keys():
                output_feature_path = output_features_path.joinpath(feature_name)
                output_feature_path.mkdir(parents=True, exist_ok=True)

                for filename, feature in output_features[feature_name].items():
                    np.save(
                        str(output_feature_path.joinpath(filename + ".npy")), feature
                    )

        del input_corpus


__all__ = ["FeaturesProcessor"]
