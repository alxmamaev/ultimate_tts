from typing import Callable, List
from russian_g2p.Transcription import Transcription as g2p_model

from ...data.corpus import Corpus, Symbol, Token, Transcription


class G2PTransform(Callable):
    def __init__(self):
        self.g2p_model = g2p_model()

    def __call__(self, corpus: Corpus) -> Corpus:
        processed_corpus = Corpus()
        for transcription in corpus:
            words = [str(token) for token in transcription]
            g2p_output = self.g2p_model.transcribe(words)

            tokens = []

            for token_indx, token in enumerate(transcription):
                token_transcription = g2p_output[token_indx]

                if not g2p_output[token_indx]:
                    # Skip token if no transcription for this token
                    continue
                else:
                    # Token may have many variants of transcription
                    # We are getting the first, as an heuristic
                    token_transcription = token_transcription[0]

                graphemes = token.graphemes
                phones = [Symbol(phone_mark) for phone_mark in token_transcription]

                tokens.append(Token(graphemes=graphemes, phones=phones))
            if not tokens:
                continue

            processed_corpus.add(
                Transcription(
                    transcription.filename,
                    speaker_id=transcription.speaker_id,
                    tokens=tokens
                )
            )

        return processed_corpus


__all__ = ["LowerCaseCleaner", "VocabCleaner"]
