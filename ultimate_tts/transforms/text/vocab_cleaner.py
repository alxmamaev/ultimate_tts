from typing import Callable, List

from ...data.corpus import Corpus, Symbol, Token, Transcription


class VocabCleanerTransform(Callable):
    """Callable cleaner thats remove all graphemes from corpus transcriptions, thats are not contains in vocab"""

    def __init__(self, vocab: List[str]) -> None:
        """Initialization function

        Parameters
        ----------
        vocab : List[str]
            List of allowed graphemes
        """
        self._vocab = set(vocab)

    def __call__(self, corpus: Corpus) -> Corpus:
        """Removed graphemes thats not contained in vocab

        Parameters
        ----------
        corpus : Corpus
            Input batch of texts

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

                    if mark in self._vocab:
                        graphemes.append(Symbol(mark, time))

                if not graphemes:
                    continue

                tokens.append(Token(graphemes=graphemes, phones=phones))

            processed_corpus.add(
                Transcription(
                    transcription.filename,
                    speaker_id=transcription.speaker_id,
                    tokens=tokens
                )
            )

        return processed_corpus
