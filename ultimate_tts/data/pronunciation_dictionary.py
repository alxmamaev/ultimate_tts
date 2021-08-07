from pathlib import Path


class PronunciationDictionary:
    def __init__(self):
        self.__dictionary = {}

    def __len__(self):
        return len(self.__dictionary)

    def keys(self):
        return self.__dictionary.keys()

    def values(self):
        return self.__dictionary.values()

    def items(self):
        return self.__dictionary.items()

    @classmethod
    def from_file(cls, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        pronunciation_dictionary = cls()
        with filepath.open("r") as f:
            for line in f:
                line = line.strip()
                token, phones = line.split(" ", 1)
                phones = phones.split(" ")

                pronunciation_dictionary.add(token, phones)

        return pronunciation_dictionary

    def add(self, token, phones):
        if token not in self.__dictionary:
            self.__dictionary[token] = phones

    def get(self, token):
        return self.__dictionary.get(token)

    def to_file(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        sorted_tokens = sorted(list(self.__dictionary.keys()))
        with filepath.open("w") as f:
            for token in sorted_tokens:
                phones = self.get(token)
                if phones is not None:
                    phones = " ".join(phones)
                    f.write(f"{token} {phones}\n")
