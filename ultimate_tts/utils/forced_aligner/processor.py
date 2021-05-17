import subprocess
import subprocess
from pathlib import Path


class MontrealForcedAlignerProcessor:
    def __init__(self):
        pass

    def process_files(self, inputs, outputs, verbose=False):
        vocab_path = inputs["vocab_path"]
        output_corpus_path = Path(outputs["corpus_path"])
        output_textgrids = Path(outputs["textgrids_path"])

        with open(inputs["metadata_path"]) as f:
            for line in f:
                filename, text = line.split("|")[:2]
                text = text.replace(" ", "").replace("\t", " ")

                output_lab_path = output_corpus_path.joinpath(filename + ".lab")
                with open(str(output_lab_path), "w") as f:
                    f.write(text)

        subprocess.call(["mfa", "train", str(output_corpus_path), str(vocab_path), str(output_textgrids)])