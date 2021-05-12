class LowerCaseCleaner:
    def __init__(self):
        pass

    def __call__(self, texts_batch):
        processed_batch = []

        for text in texts_batch:
            text = text.lower()
            processed_batch.append(text)

        return processed_batch


class VocabCleaner:
    def __init__(self, vocab):
        self.vocab = list(vocab)

    def __call__(self, texts_batch):
        processed_batch = []

        for text in texts_batch:
            text = "".join(i for i in text if i in self.vocab)
            processed_batch.append(text)

        return processed_batch


__all__ = ["LowerCaseCleaner", "VocabCleaner"]