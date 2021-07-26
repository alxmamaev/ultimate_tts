from russian_g2p.Transcription import Transcription

# FIX IT
import warnings
warnings.filterwarnings("ignore")


# In the future I will replace this module to more efficient and better g2p
# But now thats implementation not working in a batch mode and has got a bad accuracy

class RussianG2P:
    def __init__(self, word_separator="\t"):
        self.transcriptor = Transcription()
        self.word_separator = word_separator


    def __call__(self, graphemes_batch, return_word_to_phonemes_dictionary=False):
        processed_batch = []
        word_to_phonemes_dictionary = {}

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
            
            g2p_output = self.transcriptor.transcribe(words)
            for word, word_phonemes in zip(words, g2p_output):
                
                if not word_phonemes:
                    continue

                # Some words have different variant to pronounce,
                # We are selecting one of them
                word_phonemes = word_phonemes[0]
                phonemes += word_phonemes + [self.word_separator]
                
                if return_word_to_phonemes_dictionary:
                    word_to_phonemes_dictionary[word] = word_phonemes
            
            processed_batch.append(phonemes)

        if return_word_to_phonemes_dictionary:
            return processed_batch, word_to_phonemes_dictionary 
        
        return processed_batch


__all__ = ["RussianG2P"]