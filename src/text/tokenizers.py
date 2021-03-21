class CharTokenizer:
    def __init__(self, vocab):
        self.id2char = {}
        self.char2id = {}

        for i, c in enumerate(vocab):
            # zero id is padding token
            self.id2char[i + 1] = c
            self.char2id[c] = i + 1

    def text_to_sequence(self, text):
        sequence = list(text)

        return sequence

    def sequence_to_ids(self, sequence):
        sequence_ids = []

        for c in sequence:
            sequence_ids.append(self.char2id[c])
        
        return sequence_ids

    def ids_to_sequence(self, sequence_ids):
        sequence_chars = []

        for i in sequence_ids:
            sequence_chars.append(self.id2char[i])

        return sequence_chars

    def sequence_to_text(self, sequence):
        return "".join(sequence)