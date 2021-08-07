from typing import Callable, List
from ...data.corpus import Corpus, Symbol, Token, Transcription

# FIX IT
import warnings

warnings.filterwarnings("ignore")


class LowerCaseTransform(Callable):
    """Callable cleaner thats lowercase input texts"""

    def __init__(self) -> None:
        return

    def __call__(self, corpus: Corpus) -> Corpus:
        """Lowercase batch all graphemes in the batch

        Parameters
        ----------
        texts_batch : Corpus
            Input corpus of transcriptions

        Returns
        -------
        Corpus
            Returns processed corpus of transcriptions
        """
        processed_corpus = Corpus()

        for transcription in corpus:
            tokens = []
            for token in transcription:
                graphemes = []
                phones = token.phones

                for grapheme in token.graphemes:
                    mark = grapheme.mark
                    time = grapheme.time
                    mark = mark.lower()

                    graphemes.append(Symbol(mark, time))

                tokens.append(Token(graphemes=graphemes, phones=phones))

            processed_corpus.add(
                Transcription(
                    transcription.filename,
                    speaker_id=transcription.speaker_id,
                    tokens=tokens
                )
            )

        return processed_corpus
