import torch

from src.tokenizer import Tokenizer


@torch.inference_mode()
def generate(model, images, max_letters=45):
    if not 1 <= max_letters <= 45:
        raise ValueError("max_letters must be between 1 and 45")
    model.eval()
    visual = model.visual_tokens(images)
    ids = torch.empty(images.shape[0], 0, dtype=torch.long, device=images.device)
    finished = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
    for _ in range(max_letters):
        next_ids = model.decode_tokens(visual, ids)[:, -1].argmax(dim=-1)
        next_ids = torch.where(finished, Tokenizer.eos_id, next_ids)
        ids = torch.cat((ids, next_ids[:, None]), dim=1)
        finished = finished | (next_ids == Tokenizer.eos_id)
        if finished.all():
            break
    return [Tokenizer().decode(row) for row in ids.cpu()]
