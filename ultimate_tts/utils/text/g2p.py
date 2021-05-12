import re
from russian_g2p.Transcription import Transcription

WORD_PATTEN = re.compile("([\w]+)")

# FIX IT
import warnings
warnings.filterwarnings("ignore")


# In future i will replace it to more efficient and better g2p
# But now thats implementation not working in batch mode and has a bad accuracy

class RussianG2P:
    def __init__(self, word_separator="\t"):
        self.transcriptor = Transcription()
        self.word_separator = word_separator


    def __call__(self, graphemes_batch, return_vocab=False):
        processed_batch = []
        vocab = {}

        for graphemes in graphemes_batch:
            words = []
            phonemes = []
            word = ""

            for grapheme in graphemes + [self.word_separator]:
                if grapheme == self.word_separator:
                    if word.isalpha():
                        words.append(word)
                    word = ""
                else:
                    word += grapheme
            
            words_phonemes = self.transcriptor.transcribe(words)
            for word, word_phonemes in zip(words, words_phonemes):
                word_phonemes = word_phonemes[0]
                phonemes += word_phonemes + [self.word_separator]
                
                if return_vocab:
                    vocab[word] = word_phonemes
            
            processed_batch.append(phonemes)

        if return_vocab:
            return processed_batch, vocab 
        
        return processed_batch


__all__ = ["RussianG2P"]