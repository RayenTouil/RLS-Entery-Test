# FR : Génère le mot sans utiliser les lettres de la bonne réponse.
# EN: Generates a word without using letters from the correct answer.

import torch

from src.tokenizer import Tokenizer


# FR : La génération n’a pas besoin de calculer les gradients.
# EN: Generation does not need gradients.
@torch.inference_mode()
def generate(model, images, max_letters=45):
    if not 1 <= max_letters <= 45:
        raise ValueError("max_letters must be between 1 and 45")
    model.eval()
    # FR : Le CNN est calculé une seule fois pour le lot d’images.
    # EN: The CNN is computed once for the image batch.
    visual = model.visual_tokens(images)
    # FR : Commence sans lettre : le dernier token visuel donne la première prédiction.
    # EN: Starts without letters: the last visual token gives the first prediction.
    ids = torch.empty(images.shape[0], 0, dtype=torch.long, device=images.device)
    finished = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
    for _ in range(max_letters):
        # FR : Argmax choisit le token au score maximal : décodage glouton.
        # EN: Argmax picks the highest-scoring token: greedy decoding.
        next_ids = model.decode_tokens(visual, ids)[:, -1].argmax(dim=-1)
        # FR : Une séquence déjà terminée reste sur eos pendant que les autres continuent.
        # EN: A finished sequence stays at eos while the others continue.
        next_ids = torch.where(finished, Tokenizer.eos_id, next_ids)
        ids = torch.cat((ids, next_ids[:, None]), dim=1)
        finished = finished | (next_ids == Tokenizer.eos_id)
        # FR : Arrête dès que tous les mots sont terminés ; sinon limite de 45 lettres.
        # EN: Stops when every word is finished; otherwise caps the output at 45 letters.
        if finished.all():
            break
    return [Tokenizer().decode(row) for row in ids.cpu()]
