import enum
import re
from catalyst.registry import REGISTRY
from pathlib import Path
from tqdm import tqdm
from math import ceil

WORD_PATTEN = re.compile("([\w]+)")

class TextProcessor:
    def __init__(self, vocab, cleaners=[], g2p=None, words_separator="\t", batch_size=1):
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
        for cleaner in self.cleaners:
            texts_batch = cleaner(texts_batch)

        graphemes_batch = [self.__text_to_graphemes(text) for text in texts_batch]
        phonemes_batch, vocab = self.g2p(graphemes_batch, return_vocab=True) if self.g2p is not None else ([], {})

        return {"graphemes": graphemes_batch, "phonemes": phonemes_batch, "vocab": vocab}


    def __sequence_to_ids(self, sequence):
        ids_sequence = []

        for token in sequence:
            ids_sequence.append(self.token2id[token])

        return ids_sequence


    def __call__(self, texts_batch, return_ids=True):
        processed_batch = []
        if self.g2p is None:
            processed_batch = self.__process_texts_batch(texts_batch)["graphemes"]
        else:
            processed_batch = self.__process_texts_batch(texts_batch)["phonemes"]

        
        if return_ids:
            return [self.__sequence_to_ids(sequence) for sequence in processed_batch]
        else:
            return processed_batch


    def process_files(self, inputs, outputs, verbose=False):
        input_metadata = []
        output_metadata = []
        output_vocab = {}

        input_metadata_path = Path(inputs["metadata_path"])
        output_metadata_path = Path(outputs["metadata_path"])
        output_vocab_path = Path(outputs["vocab_path"]) if outputs.get("vocab_path", False) else None

        for i in [output_metadata_path, output_vocab_path]:
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
            
            output_vocab.update(processed_batch["vocab"])

            if verbose:
                progress_bar.update(1)
            
        
        with open(output_metadata_path, "w", encoding="utf-8") as f:
            for filename, graphemes, phonemes in output_metadata:
                graphemes = " ".join(graphemes)
                phonemes = " ".join(phonemes)

                f.write(f"{filename}|{graphemes}|{phonemes}\n")

        if output_vocab_path is not None:
            with open(output_vocab_path, "w", encoding="utf-8") as f:
                for word in sorted(output_vocab.keys()):
                    phonemes = " ".join(output_vocab[word])
                    word = word.upper()
                    f.write(f"{word} {phonemes}\n")


__all__ = ["TextProcessor"]