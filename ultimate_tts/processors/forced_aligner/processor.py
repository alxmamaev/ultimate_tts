import logging
import os
import subprocess
from pathlib import Path
from typing import Dict

import torchaudio as taudio
from textgrid import TextGrid
from textgrid.exceptions import TextGridError
from tqdm import tqdm

from ...data.corpus import Corpus, Symbol, Time
from ...utils.functional import align_symbols_to_transcription


class MontrealForcedAlignerProcessor:
    """Processor wraps Montreal Forced Aligner, for align text with audio.
    That's processor do few steps:
    * reads metadata and creates corpus for MFA, that is directory that contains audio in wav format
    and text trascription in .lab format.
    * gets path to dictionary file, thats contains transcription for every word in the text
    * runs MFA training and aligning subprocess
    """

    def __init__(self):
        self.UNKNOWN_TOKEN_NAME = "spn"
        self.SCILENCE_TOKEN_MARK = "sil"
        return

    def __call__(self):
        raise NotImplementedError()

    def process_files(self, inputs: Dict, outputs: Dict, verbose: bool = False) -> None:
        logging.info("Start forced aligner processor processor")

        input_corpus_path = Path(inputs["corpus_path"])
        input_wavs_path = Path(inputs["wavs_path"])
        output_pronounce_dictionary_path = outputs["pronounce_dictionary"]
        output_textgrids = Path(outputs["textgrids_path"])
        output_wavs_path = Path(outputs["wavs_path"])
        output_corpus_path = Path(outputs["corpus_path"])

        output_wavs_path.mkdir(parents=True, exist_ok=True)
        output_textgrids.mkdir(parents=True, exist_ok=True)

        logging.info("Read corpus data")
        input_corpus = Corpus.from_file(input_corpus_path)

        logging.info("Dump lab files")
        input_corpus.dump_labs(input_wavs_path)

        logging.info("create pronounce dictionary")
        pronounce_dictionary = input_corpus.get_pronounce_dictionary()
        pronounce_dictionary.to_file(output_pronounce_dictionary_path)

        logging.info("Start forced aligner")
        ncpu = os.cpu_count()
        # subprocess.call(
        #     [
        #         "mfa",
        #         "train",
        #         str(input_wavs_path),
        #         str(output_pronounce_dictionary_path),
        #         str(output_textgrids),
        #         "-j",
        #         str(ncpu),
        #         "-c",
        #     ]
        # )

        logging.info("Search TextGrid output files")
        filename_to_textgrid_path = {}

        for textgrid_file_path in list(output_textgrids.glob("*.TextGrid")):
            filename = textgrid_file_path.stem.rsplit("-", 1)[-1]
            filename_to_textgrid_path[filename] = textgrid_file_path

        aligned_corpus = Corpus()


        logging.info("Process textgrid output files")

        for transcription in tqdm(input_corpus):
            filename = transcription.filename

            if filename not in filename_to_textgrid_path:
                logging.warning(f"Skip file {filename}, no TextGrid file (probably mfa can't create alignment for this file)")
                continue

            textgrid_file_path = filename_to_textgrid_path[filename]

            try:
                textgrid = TextGrid.fromFile(str(textgrid_file_path))
            except TextGridError:
                logging.warning(f"Skip file {filename}, corrupted TextGrid file")
                continue

            # Reading phones
            phones = []
            for grid in textgrid:
                if grid.name == "phones":
                    for phone_alignment in grid:
                        if (
                            phone_alignment.mark == ""
                            or phone_alignment.mark == self.SCILENCE_TOKEN_MARK
                        ):
                            continue

                        if phone_alignment.mark == self.UNKNOWN_TOKEN_NAME:
                            phones = []
                            break

                        time = Time(phone_alignment.minTime, phone_alignment.maxTime)
                        phone = Symbol(phone_alignment.mark, time=time)
                        phones.append(phone)

                    break

            if not phones:
                logging.warning(f"Skip file {filename}, bad alignment")
                continue

            # Trim audio
            audio_start = phones[0].time.start
            audio_end = phones[-1].time.end

            input_wav_path = input_wavs_path.joinpath(filename + ".wav")
            wav, rate = taudio.load(str(input_wav_path))

            start_indx = int(audio_start * rate)
            end_indx = int(audio_end * rate)

            wav = wav[:, start_indx:end_indx]
            phones_ = phones
            phones = []

            for phone in phones_:
                time = Time(
                    phone.time.start - audio_start, phone.time.end - audio_start
                )
                phones.append(Symbol(phone.mark, time=time))
            # del phones_

            output_wav_path = output_wavs_path.joinpath(filename + ".wav")
            taudio.save(str(output_wav_path), wav, rate)

            aligned_transcription = align_symbols_to_transcription(symbols=phones, 
                                                                   transcription=transcription, 
                                                                   align_by_phones=True)
            if not aligned_transcription:
                logging.warning(f"Skip file {filename}, bad alignment")
                aligned_corpus.add(aligned_transcription)

        aligned_corpus.to_file(output_corpus_path)


__all__ = ["MontrealForcedAlignerProcessor"]
