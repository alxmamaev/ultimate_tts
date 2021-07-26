import subprocess
from pathlib import Path
import os


class MontrealForcedAlignerProcessor:
    """Processor wraps Montreal Forced Aligner, for align text with audio.
    That's processor do few steps:
    * reads metadata and creates corpus for MFA, that is directory that contains audio in wav format
    and text trascription in .lab format.
    * gets path to dictionary file, thats contains transcription for every word in the text
    * runs MFA training and aligning subprocess
    """
    def __init__(self):
        return

    def process_files(self, inputs, outputs, verbose=False):
        vocab_path = inputs["word_to_phonemes_dictionary_path"]
        output_corpus_path = Path(outputs["corpus_path"])
        output_textgrids = Path(outputs["textgrids_path"])

        with open(inputs["metadata_path"]) as f:
            for line in f:
                filename, text = line.split("|")[:2]
                text = text.replace(" ", "").replace("\t", " ")

                output_lab_path = output_corpus_path.joinpath(filename + ".lab")
                with open(str(output_lab_path), "w") as f:
                    f.write(text)

        if verbose:
            print("Starting forced aligner")

        ncpu = os.cpu_count()
        subprocess.call(["mfa", "train", str(output_corpus_path), str(vocab_path), str(output_textgrids), "-j", str(ncpu), "-c"])