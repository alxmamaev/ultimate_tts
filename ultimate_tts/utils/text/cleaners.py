class LowerCaseCleaner:
    """Callable cleaner thats lowercase input texts
    """
    def __init__(self):
        return


    def __call__(self, texts_batch):
        """Lowercase batch all texts in batch

        Parameters
        ----------
        texts_batch : List[str]
            Input batch of texts

        Returns
        -------
        List[str]
            Returns batch of lowercased texts
        """
        processed_batch = []

        for text in texts_batch:
            text = text.lower()
            processed_batch.append(text)

        return processed_batch


class VocabCleaner:
    """Callable cleaner thats remove all charracters from texts, thats are not contains in vocab
    """
    def __init__(self, vocab):
        """Initialization function

        Parameters
        ----------
        vocab : List[str]
            List of allowed charracters
        """
        self.vocab = list(vocab)


    def __call__(self, texts_batch):
        """Removed charracters thats not contained in vocab

        Parameters
        ----------
        texts_batch : List[str]
            Input batch of texts

        Returns
        -------
        List[str]
            Returns batch of texts thats not contains not allowed characters
        """
        processed_batch = []

        for text in texts_batch:
            text = "".join(i for i in text if i in self.vocab)
            processed_batch.append(text)

        return processed_batch


__all__ = ["LowerCaseCleaner", "VocabCleaner"]