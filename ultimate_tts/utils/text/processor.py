import enum
import re
from catalyst.registry import REGISTRY
from pathlib import Path
from tqdm import tqdm
from math import ceil

WORD_PATTEN = re.compile("([\w]+)")


class TextProcessor:
    """TextProcessor handling text processing, like text cleaning, tokenization and g2p.
    """
    def __init__(self, vocab, cleaners=[], g2p=None, words_separator="\t", batch_size=1):
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
        self.vocab = vocab
        self.words_separator = words_separator
        self.batch_size = batch_size

        self.token2id = {}

        # zero token id for padding
        for i, token in enumerate(self.vocab, 1):
            self.token2id[token] = i

        self.cleaners = []

        for cleaner in cleaners:
            if isinstance(cleaner, dict):
                cleaner = REGISTRY.get_from_params(**cleaner)

            self.cleaners.append(cleaner)

        if isinstance(g2p, dict):
            g2p = REGISTRY.get_from_params(**g2p)

        self.g2p = g2p


    def __text_to_graphemes(self, text):
        """Convert input text into list of graphemes (alphabet characters), 
        where every word is separated by word separator token.

        Parameters
        ----------
        text : str
            Input cleaned text.

        Returns
        -------
        List[str]
            Output list of graphemes.
        """
        graphemes = []
        words = re.split(WORD_PATTEN, text)
        for i, word in enumerate(words):
            word = word.strip()
            if not word:
                continue

            graphemes += list(word) + [self.words_separator]
        
        # Remove last word separator at the end of sentence
        return graphemes[:-1]


    def __process_texts_batch(self, texts_batch):
        """Applies processor cleaners to texts batch and apply g2p if that defined.

        Parameters
        ----------
        texts_batch : List[str]
            Input batch of texts.

        Returns
        -------
        Dict
            Return processed batch of following structure:
            {
                "graphemes", [... , ...] - List of string, graphemes representations of every text in batch
                "phonemes", [... , ...] - List of string, phonemes representations of every text, if g2p defined, emty list otherwise
                "word_to_phonemes_dictionary": {... : ... , ... : ...} - Dict represents a phonemes transcription for every unique word in the texts,
                                                                         if g2p defined, empty dict othervise
            }
        """

        for cleaner in self.cleaners:
            texts_batch = cleaner(texts_batch)

        graphemes_batch = [self.__text_to_graphemes(text) for text in texts_batch]
        if self.g2p is not None:
            phonemes_batch, word_to_phonemes_dictionary = self.g2p(graphemes_batch, return_word_to_phonemes_dictionary=True) 
        else:
            phonemes_batch, word_to_phonemes_dictionary = [], {}

        return {"graphemes": graphemes_batch, "phonemes": phonemes_batch, "word_to_phonemes_dictionary": word_to_phonemes_dictionary}


    def __sequence_to_ids(self, sequence):
        """Converts sequence of tokens to sequence of their ids, according to the processor vocab

        Parameters
        ----------
        sequence : List[str]
            List of input tokens

        Returns
        -------
        List[int]
            List of input tokens ids
        """
        ids_sequence = []

        for token in sequence:
            ids_sequence.append(self.token2id[token])

        return ids_sequence


    def __call__(self, texts_batch, return_ids=True):
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
        processed_batch = []
        if self.g2p is None:
            processed_batch = self.__process_texts_batch(texts_batch, return_word_to_phonemes_dictionary=False)["graphemes"]
        else:
            processed_batch = self.__process_texts_batch(texts_batch)["phonemes"]

        
        if return_ids:
            return [self.__sequence_to_ids(sequence) for sequence in processed_batch]
        else:
            return processed_batch


    def process_files(self, inputs, outputs, verbose=False):
        input_metadata = []
        output_metadata = []
        output_word_to_phonemes_dictionary = {}

        input_metadata_path = Path(inputs["metadata_path"])
        output_metadata_path = Path(outputs["metadata_path"])
        output_word_to_phonemes_dictionary_path = Path(outputs["word2phone_dictionary_path"]) if outputs.get("word2phone_dictionary_path", False) else None

        for i in [output_metadata_path, output_word_to_phonemes_dictionary_path]:
            if i is not None:
                i.parent.mkdir(parents=True, exist_ok=True)


        with open(input_metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                filename, text = line.split("|")[:2]
                input_metadata.append((filename, text))


        item_index = 0
        if verbose:
            print("Processing texts...")
            progress_bar = tqdm(total=int(ceil(len(input_metadata) / self.batch_size)))

        while True:
            texts_batch = []
            filenames_batch = []

            while item_index < len(input_metadata) and len(filenames_batch) < self.batch_size:
                filename, text = input_metadata[item_index]
                filenames_batch.append(filename)
                texts_batch.append(text)
                item_index += 1
            
            if not filenames_batch:
                break

            processed_batch = self.__process_texts_batch(texts_batch)

            for i, filename in enumerate(filenames_batch):
                output_metadata.append((filename, 
                                       processed_batch["graphemes"][i],
                                       processed_batch["phonemes"][i]))

            output_word_to_phonemes_dictionary.update(processed_batch["word_to_phonemes_dictionary"])

            if verbose:
                progress_bar.update(1)

        
        with open(output_metadata_path, "w", encoding="utf-8") as f:
            for filename, graphemes, phonemes in output_metadata:
                graphemes = " ".join(graphemes)
                phonemes = " ".join(phonemes)

                f.write(f"{filename}|{graphemes}|{phonemes}\n")


        if output_word_to_phonemes_dictionary_path is not None:
            with open(output_word_to_phonemes_dictionary_path, "w", encoding="utf-8") as f:
                for word in sorted(output_word_to_phonemes_dictionary.keys()):
                    phonemes = " ".join(output_word_to_phonemes_dictionary[word])
                    word = word.upper()
                    f.write(f"{word} {phonemes}\n")


__all__ = ["TextProcessor"]