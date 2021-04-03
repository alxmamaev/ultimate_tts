class LowerCaseCleaner:
    def __init__(self):
        pass

    def __call__(self, text):
        text = text.lower()

        return text


class VocabCleaner:
    def __init__(self, vocab):
        self.vocab = list(vocab)

    def __call__(self, text):
        text = "".join(i for i in text if i in self.vocab)

        return text


__all__ = ["LowerCaseCleaner", "VocabCleaner"]