import enum
from os import PathLike
from pathlib import Path
from copy import copy
import json
from typing import List, Union
from .pronunciation_dictionary import PronunciationDictionary


class OutputType(enum.Enum):
    PHONES = enum.auto()
    GRAPHEMES = enum.auto()


class Time:
    def __init__(self, start, end):
        assert isinstance(start, float), "Start time must be float"
        assert isinstance(end, float), "End time must be float"

        self._start = start
        self._end = end

    def __repr__(self):
        return f"({self._start}, {self._end})"

    def __eq__(self, other) -> bool:
        return other.start == self.start and other.end == self.end

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    def to_dict(self):
        data = {"start": self._start, "end": self._end}

        return data

    def from_dict(cls, data):
        start = data["start"]
        end = data["end"]

        return cls(start, end)


class Symbol:
    def __init__(self, mark, time=None):
        assert isinstance(mark, str), "Mark must be string"
        assert (
            isinstance(time, Time) or time is None
        ), f"Mark must is instance of {Time} or None"

        self._mark = mark
        self._time = time

    def __repr__(self):
        return f'Symbol(mark="{self._mark}", time={self._time})'

    def __str__(self):
        return self._mark

    @property
    def mark(self):
        return self._mark

    @property
    def time(self):
        return self._time

    def to_dict(self):
        if self._time is None:
            time_dict = None
        else:
            time_dict = self._time.to_dict()

        data = {"mark": self._mark, "time": time_dict}

        return data

    @classmethod
    def from_dict(cls, data):
        mark = data["mark"]
        time_data = data["time"]
        if time_data is None:
            time = None
        else:
            time = Time.from_dict(time_data)

        return cls(mark, time)


class Token:
    def __init__(self, graphemes, phones=None):
        if isinstance(graphemes, str):
            graphemes = self.string_to_graphemes(graphemes)

        phones = phones or []

        graphemes_hase_time = [g.time is not None for g in graphemes]
        phones_hase_time = [p.time is not None for p in phones]


        if graphemes:
            assert all(graphemes_hase_time) == any(
                graphemes_hase_time
            ), "Some graphemes has not a time, you must to set a time for all graphemes, or no one"

        if phones:
            assert all(phones_hase_time) == any(
                phones_hase_time
            ), "Some phones has not a time, you must to set a time for all phones, or no one"

        self._time = None

        if all(graphemes_hase_time) and graphemes:
            previos_end_time = 0

            for grapheme in graphemes:
                assert (
                    previos_end_time <= grapheme.time.start
                ), "Grapheme {grapheme} starts before than previos grapheme ends"
                previos_end_time = grapheme.time.end

            self._time = Time(
                graphemes[0].time.start, graphemes[-1].time.end
            )

        self._graphemes = copy(graphemes)

        if all(phones_hase_time) and phones:
            previos_end_time = 0

            for phone in phones:
                assert (
                    previos_end_time <= phone.time.start
                ), "Phone {phone} starts before than previos phone ends"
                previos_end_time = phone.time.end

            if self._time is None:
                self._time = Time(phones[0].time.start, phones[-1].time.end)
            else:
                assert (
                    self._time.start == phones[0].time.start
                ), "Start time of phones and graphemes is not equal"
                assert (
                    self._time.end == phones[-1].time.end
                ), "End time of phones and graphemes is not equal"

        self._phones = copy(phones)

    def __str__(self):
        mark = "".join([grapheme.mark for grapheme in self.graphemes])
        return mark

    def __repr__(self):
        graphemes = "".join([str(grapheme) for grapheme in self.graphemes])
        phones = "".join([str(phone) for phone in self.phones])

        return f'Token(graphemes="{graphemes}", phones="{phones}")'

    @staticmethod
    def string_to_graphemes(mark):
        graphemes = []
        for c in mark:
            grapheme = Symbol(c)
            graphemes.append(grapheme)

        return graphemes

    @property
    def phones(self) -> List[Symbol]:
        phones = self._phones
        return phones

    @property
    def graphemes(self) -> List[Symbol]:
        graphemes = self._graphemes
        return graphemes
        
    @property
    def time(self) -> Time:
        return self._time

    def to_dict(self) -> dict:
        data = {
            "graphemes": [grapheme.to_dict() for grapheme in self._graphemes],
            "phones": [phone.to_dict() for phone in self._phones],
        }

        return data

    @classmethod
    def from_dict(cls, data: dict):
        graphemes = [
            Symbol.from_dict(grapheme_dict) for grapheme_dict in data["graphemes"]
        ]
        phones = [Symbol.from_dict(phone_dict) for phone_dict in data["phones"]]

        return cls(graphemes, phones)


class Transcription:
    def __init__(self, filename, speaker_id=None, tokens=None):
        self._filename = filename
        self._tokens = []
        self._speaker_id = speaker_id
        self._time = None

        tokens = tokens or []
        tokens_has_time = [t.time is not None for t in tokens]

        if tokens:
            assert all(tokens_has_time) == any(
                tokens_has_time
            ), "Some tokens has not a time, you must to set a time for all tokens, or no one"


            if all(tokens_has_time):
                previos_end_time = 0

                for token in tokens:
                    assert (
                        previos_end_time <= token.time.start
                    ), "Grapheme {grapheme} starts before than previos grapheme ends"
                    previos_end_time = token.time.end

                self._time = Time(
                    tokens[0].time.start, tokens[-1].time.end
                )

        self._tokens = copy(tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __bool__(self) -> bool:
        return bool(self._tokens)

    def __str__(self) -> str:
        text = " ".join(str(token) for token in self._tokens)
        return text

    def __repr__(self) -> str:
        tokens_repr = ", ".join([repr(token) for token in self._tokens])
        return f"Transcription(filename={self._filename}, speaker_id={self._speaker_id}, tokens=[{tokens_repr}]"

    def __getitem__(self, index) -> Token:
        token = self._tokens[index]
        return token

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def speaker_id(self) -> int:
        return self._speaker_id

    @property
    def time(self) -> Time:
        return self._time

    def to_dict(
        self,
    ) -> dict:
        data = {
            "tokens": [token.to_dict() for token in self._tokens],
            "filename": self._filename,
            "speaker_id": self._speaker_id,
        }

        return data

    def get_symbols(self, pause_mark="<PAUSE>", retrun_phones=True):
        symbols = []

        prev_symbol_time = Time(0, 0) if self.time is not None else None

        for token in self._tokens:
            if retrun_phones:
                token_symbols = token.phones
            else:
                token_symbols = token.graphemes

            for symbol in token_symbols:
                if self.time is not None and symbols.time.start - prev_symbol_time.end > 0:
                    symbols.append(
                        Symbol(
                            pause_mark,
                            time=Time(prev_symbol_time.end, symbol.time.start),
                        )
                    )

                symbols.append(symbol)
                prev_symbol_time = symbol.time

            if not self._has_time:
                symbols.append(Symbol(pause_mark, time=None))

        return symbols

    @classmethod
    def from_dict(cls, data: dict):
        filename = data["filename"]
        speaker_id = data["speaker_id"]
        tokens = [Token.from_dict(token_dict) for token_dict in data["tokens"]]

        return cls(filename, speaker_id, tokens)


class Corpus:
    def __init__(self, transcriptions: Union[None, List[Transcription]] = None):
        self._transcriptions = transcriptions or []

    def __len__(self):
        return len(self._transcriptions)

    def __getitem__(self, key):
        if isinstance(key, slice):
            corpus = Corpus(self._transcriptions[key])
            return corpus
        else:
            transcription = self._transcriptions[key]
            return transcription

    @classmethod
    def from_file(cls, filepath: Union[str, Path]):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        corpus = cls()

        with filepath.open("r") as f:
            for line in f:
                transcription_json = line.strip()
                if not transcription_json:
                    continue

                transcription_dict = json.loads(transcription_json)
                transcription = Transcription.from_dict(transcription_dict)
                corpus.add(transcription)

        return corpus

    def add(self, trascription: Transcription) -> None:
        assert isinstance(
            trascription, Transcription
        ), f"input transcription must be instance of {Transcription}"
        self._transcriptions.append(trascription)

    def update(self, other_corpus) -> None:
        for transciption in other_corpus:
            self.add(transciption)

    def to_file(self, filepath: Union[str, Path]):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as f:
            for transcription in self._transcriptions:
                transcription_dict = transcription.to_dict()
                transcription_json = json.dumps(transcription_dict, ensure_ascii=False)
                f.write(transcription_json + "\n")

    def dump_labs(self, distpath: Union[str, Path]):
        if not isinstance(distpath, Path):
            distpath = Path(distpath)

        distpath.parent.mkdir(parents=True, exist_ok=True)
        for transcription in self._transcriptions:
            filepath = distpath.joinpath(transcription.filename + ".lab")

            with filepath.open("w") as f:
                f.write(str(transcription))

    def get_pronounce_dictionary(self) -> PronunciationDictionary:
        pronounce_dictionary = PronunciationDictionary()

        for transcription in self._transcriptions:
            for token in transcription:
                graphemes = str(token)
                phones = [str(i) for i in token.phones]

                if not phones:
                    continue

                pronounce_dictionary.add(graphemes, phones)

        return pronounce_dictionary
