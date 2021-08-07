from math import ceil
from typing import List

import torch
from ultimate_tts.data.corpus import Symbol, Token, Transcription


def align_symbols_to_transcription(
    symbols: List[Symbol], transcription: Transcription, align_by_phones: bool = True
) -> Transcription:
    transcription_symbols = []
    transcription_tokens_indedexes = []

    for token_index, token in enumerate(transcription):
        if align_by_phones:
            token_symbols = token.phones
        else:
            token_symbols = token.graphemes

        for symbol in token_symbols:
            transcription_symbols.append(symbol)
            transcription_tokens_indedexes.append(token_index)

    if not symbols or not transcription_symbols:
        # If we have no tokens for aligning
        # Return the empty transcription
        return Transcription(
            transcription.filename, transcription.speaker_id, tokens=None
        )

    # Modified Knuth-Moris-Praat algorithm
    # for matching input symbols with symbols at transcription
    prefix_len = [0] * len(transcription_symbols)
    prefix_begin = None

    if transcription_symbols[0].mark == symbols[0].mark:
        prefix_len[0] = 1

    for symbol_index in range(1, len(transcription_symbols)):
        prev_prefix_len = prefix_len[symbol_index - 1]
        previos_symbol = transcription_symbols[symbol_index - 1]
        current_symbol = transcription_symbols[symbol_index]

        if prev_prefix_len == 0 and previos_symbol.mark == current_symbol.mark:
            # We are skip alignment step there, because we cannot align only part of the token.
            # We may align only all tokens phonemes, or no one.
            continue

        if symbols[prev_prefix_len].mark == current_symbol.mark:
            prefix_len[symbol_index] = prev_prefix_len + 1

        if prefix_len[symbol_index] == len(symbols):

            if (
                symbol_index != len(transcription_symbols) - 1
                and transcription_symbols[symbol_index].token_id
                == transcription_symbols[symbol_index + 1].token_id
            ):
                # If the end symbol of prefix is not a last symbol of the some token - we are skip this alignment
                prefix_len[symbol_index] = 0
            else:
                # Othervise, we found the alignment
                prefix_begin = symbol_index - prefix_len[symbol_index] + 1
                break

    if prefix_begin is None:
        # If we not found alignment return the empty transctiption
        return Transcription(
            transcription.filename, transcription.speaker_id, tokens=None
        )

    aligned_tokens = []
    token_phones_ = []
    token_index = transcription_tokens_indedexes[prefix_begin]

    for symbol_index, symbol in enumerate(symbols):
        current_token_index = transcription_tokens_indedexes[symbol_index]

        if current_token_index == token_index:
            token_phones_.append(symbol)
        else:
            aligned_tokens.append(
                Token(
                    graphemes=transcription[token_index].graphemes, phones=token_phones_
                )
            )

            token_index = current_token_index
            token_phones_ = [symbol]

    if token_phones_:
        aligned_tokens.append(
            Token(transcription[token_index].graphemes, token_phones_)
        )

    aligned_transcription = Transcription(
        transcription.filename,
        speaker_id=transcription.speaker_id,
        tokens=aligned_tokens,
    )

    return aligned_transcription


def split_to_batches(data, batch_size=32):
    data_len = len(data)

    for i in range(ceil(data_len / batch_size)):
        batch = data[i * batch_size : (i + 1) * batch_size]
        yield batch


def normalize_mel(mel):
    return torch.log10(mel + 1e-3)


def denormalize_mel(mel):
    return torch.pow(10.0, mel) - 1e-3
