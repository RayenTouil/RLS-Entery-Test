# FR : Convertit les mots en nombres et prépare les séquences d'un lot.
# EN: Converts words to numbers and prepares the sequences in a batch.

import torch


class Tokenizer:
    # FR : Les 26 lettres occupent les ids 0 à 25 ; 26 termine le mot.
    # EN: Letters use ids 0 to 25; 26 marks the end of a word.
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    eos_id = 26
    vocab_size = 27
    ignore_index = -100

    # FR : Ajoute eos après les lettres pour apprendre quand arrêter la génération.
    # EN: Adds eos after the letters so the model learns when to stop.
    def encode(self, word):
        if not word or any(letter not in self.alphabet for letter in word):
            raise ValueError("A word must contain only lowercase letters a-z")
        return [self.alphabet.index(letter) for letter in word] + [self.eos_id]

    # FR : Reconstruit le texte et arrête la lecture au premier eos.
    # EN: Builds the text again and stops at the first eos.
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
        # FR : Tous les mots du lot doivent avoir la même longueur de stockage.
        # EN: All words in a batch need the same storage length.
        length = max(len(ids) for ids in sequences)
        # FR : -100 est ignoré par la loss ; ce n’est pas un token du vocabulaire.
        # EN: -100 is ignored by the loss; it is not a vocabulary token.
        targets = torch.full((len(sequences), length), self.ignore_index, dtype=torch.long)
        for row, ids in enumerate(sequences):
            targets[row, :len(ids)] = torch.tensor(ids)
        # FR : Décalage : le dernier token visuel prédit la première lettre.
        # EN: Shift: the last visual token predicts the first letter.
        inputs = targets[:, :-1].clone()
        # FR : Le padding d’entrée utilise 0, car un embedding ne peut pas lire -100.
        # EN: Input padding uses 0 because an embedding cannot read -100.
        inputs[inputs == self.ignore_index] = 0
        return inputs, targets
