import torch


class Tokenizer:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    eos_id = 26
    vocab_size = 27
    ignore_index = -100

    def encode(self, word):
        if not word or any(letter not in self.alphabet for letter in word):
            raise ValueError("A word must contain only lowercase letters a-z")
        return [self.alphabet.index(letter) for letter in word] + [self.eos_id]

    def decode(self, ids):
        letters = []
        for token in ids:
            token = int(token)
            if token == self.eos_id:
                break
            if not 0 <= token < 26:
                raise ValueError("Invalid token id")
            letters.append(self.alphabet[token])
        return "".join(letters)

    def batch(self, sequences):
        length = max(len(ids) for ids in sequences)
        targets = torch.full((len(sequences), length), self.ignore_index, dtype=torch.long)
        for row, ids in enumerate(sequences):
            targets[row, :len(ids)] = torch.tensor(ids)
        inputs = targets[:, :-1].clone()
        inputs[inputs == self.ignore_index] = 0
        return inputs, targets
