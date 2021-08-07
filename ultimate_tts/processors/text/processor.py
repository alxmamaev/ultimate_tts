import logging
import re
from math import ceil
from pathlib import Path
from typing import Callable, Dict, List, Union

from catalyst.registry import REGISTRY
from tqdm import tqdm

from ...data.corpus import Corpus
from ...utils.functional import split_to_batches


class TextProcessor:
    """TextProcessor handling text processing, like text cleaning, tokenization and g2p."""

    def __init__(
        self,
        transforms: Union[List[Callable], List[Dict]],
        batch_size: int = 1,
    ):
        """Processor initialization

        Parameters
        ----------
        vocab : List[str]
            List of all tokens, thats will be used after text processing.
            Use phonemes list if you want use g2p, or graphemes (alphabet characters) othervise
        cleaners : Union[List[Callable], List[dict]], optional
            List of cleaners callable objects, or their config dicts.
        g2p : Union[Callable, dict], optional
            g2p callable object or their config config dict.
        words_separator : str, optional
            Token thats will be separate words, by default "\t"
        batch_size : int, optional
            Batch size for data processing, by default 1
        """
        self.batch_size = batch_size

        self.transforms = []

        for transform in transforms:
            if isinstance(transform, dict):
                transform = REGISTRY.get_from_params(**transform)

            self.transforms.append(transform)

    def __call__(self, corpus: Corpus, return_ids: bool = True) -> Corpus:
        """Function applies cleaners and g2p (if defined) to all input texts in the batch, and returns
           list of tokens (graphemes or phonemes) or their ids, according to the processor vocab

        Parameters
        ----------
        texts_batch : List[str]
            Input batch of text strings
        return_ids : bool, optional
            Return tokens ids, instead of tokens, by default True

        Returns
        -------
        Union[List[int], List[str]]
            Return batch of input token ids, if return_ids=True
            batch oh tokens othervise
        """

        for transform in self.transforms:
            corpus = transform(corpus)

        return corpus

    def process_files(self, inputs: Dict, outputs: Dict, verbose: bool = True) -> None:
        logging.info("Start text processor")

        input_corpus_path = Path(inputs["corpus_path"])
        output_corpus_path = Path(outputs["corpus_path"])

        logging.info("Read corpus file")

        input_corpus = Corpus.from_file(input_corpus_path)

        batched_input_corpus = split_to_batches(
            input_corpus, batch_size=self.batch_size
        )
        batches_count = ceil(len(input_corpus) / self.batch_size)

        logging.info("Start corpus processing")
        if verbose:
            batched_input_corpus = tqdm(batched_input_corpus, total=batches_count)

        output_corpus = Corpus()

        for input_corpus_batch in batched_input_corpus:
            processed_corpus_batch = self.__call__(input_corpus_batch)
            output_corpus.update(processed_corpus_batch)

        logging.info("Save processed corpus")
        output_corpus.to_file(output_corpus_path)


__all__ = ["TextProcessor"]
