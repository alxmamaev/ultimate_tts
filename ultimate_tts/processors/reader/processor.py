import logging
from pathlib import Path
from typing import Dict, Union

from nltk.tokenize import wordpunct_tokenize

from ...data.corpus import Corpus, Token, Transcription


class LJSpeechReaderProcessor:
    def __init__(self) -> None:
        return

    def __call__(self, metadata_path: Union[str, Path]) -> Corpus:
        corpus = Corpus()

        if isinstance(metadata_path, str):
            metadata_path = Path(metadata_path)

        with metadata_path.open("r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                filename, text = line.split("|")[:2]

                tokens = []
                for token_mark in wordpunct_tokenize(text):
                    tokens.append(Token(token_mark))

                corpus.add(Transcription(filename, tokens=tokens))

        return corpus

    def process_files(self, inputs: Dict, outputs: Dict, verbose: bool = True) -> None:
        logging.info("Start reader processor")

        input_metadata_path = Path(inputs["metadata_path"])
        output_corpus_path = Path(outputs["corpus_path"])

        logging.info("Read metadata file")
        corpus = self.__call__(input_metadata_path)

        logging.info("Save corpus to file")
        corpus.to_file(output_corpus_path)
