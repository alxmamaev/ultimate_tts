class TextPreprocesser:
    def __init__(self, tokenizer, cleaners=[]):
        self.tokenizer = tokenizer
        self.cleaners = cleaners

    def __call__(self, text):
        for cleaner in self.cleaners:
            text = cleaner(text)

        sequence = self.tokenizer.text_to_sequence(text)
        tokens = self.tokenizer.sequence_to_ids(sequence)

        return tokens

    def inverse(self, tokens):
        sequence = self.tokenizer.ids_to_sequence(tokens)
        text = self.tokenizer.sequence_to_text(sequence)

        return text