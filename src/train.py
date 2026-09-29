# FR : Entraîne le modèle, mesure la loss et sauvegarde les checkpoints.
# EN: Trains the model, measures loss and saves checkpoints.

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from src.data import make_loader
from src.model.vlm import TinyVLM
from src.utils import hardware, plot_losses, read_config, setup, synchronize, write_json


def run_epoch(model, loader, device, optimizer=None):
    # FR : Sans optimiseur, cette fonction effectue seulement la validation.
    # EN: Without an optimizer, this function only runs validation.
    model.train(optimizer is not None)
    total_loss, total_tokens = 0.0, 0
    # FR : Ignore les positions ajoutées pour compléter les mots.
    # EN: Ignores positions added to pad the words.
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # FR : La validation ne construit pas de graphe de gradients.
    # EN: Validation does not build a gradient graph.
    with torch.set_grad_enabled(optimizer is not None):
        for images, inputs, targets in loader:
            # FR : Place images, entrées et cibles sur le même CPU ou GPU.
            # EN: Moves images, inputs and targets to the same CPU or GPU.
            images, inputs, targets = [x.to(device) for x in (images, inputs, targets)]
            if optimizer is not None:
                # FR : Efface les gradients du lot précédent.
                # EN: Clears gradients from the previous batch.
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, inputs)
            # FR : Aplatit B et T : un groupe de 27 logits par cible.
            # EN: Flattens B and T: one set of 27 logits per target.
            loss = criterion(logits.reshape(-1, 27), targets.reshape(-1))
            if optimizer is not None:
                # FR : Calcule les gradients, limite leur norme, puis met à jour les poids.
                # EN: Computes gradients, clips their norm, then updates the weights.
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            # FR : Pondère la moyenne par le nombre de vrais tokens, pas par le nombre de lots.
            # EN: Weights the mean by real token count, not by batch count.
            count = (targets != -100).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
    return total_loss / total_tokens


def main():
    # FR : Lit la configuration, le périphérique et un éventuel checkpoint de reprise.
    # EN: Reads the config, device and an optional checkpoint to resume.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = read_config(args.config)
    setup(config["seed"], config["threads"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or install a CUDA build")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    model = TinyVLM(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    overfit = "overfit_steps" in config
    history, best, start = [], float("inf"), 0
    # FR : Restaure aussi l’état d’AdamW pour continuer l’entraînement.
    # EN: Also restores AdamW state to continue training.
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        for key in config:
            if key not in ("epochs", "overfit_steps") and config[key] != saved["config"][key]:
                raise ValueError(f"Resume config differs at {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, start = saved["history"], saved["best"], saved["completed"]
    train_loader = make_loader(config, "train", shuffle=True)
    # FR : E0 réutilise toujours le même lot ; les vrais runs utilisent tout le train.
    # EN: E0 always reuses one batch; full runs use the entire training split.
    fixed_batch = [next(iter(train_loader))] if overfit else None
    val_loader = None if overfit else make_loader(config, "val")
    limit = config["overfit_steps"] if overfit else config["epochs"]
    run_hardware = hardware(device)
    write_json(output / "config.json", config)
    write_json(output / "hardware.json", run_hardware)
    print({"parameters": sum(p.numel() for p in model.parameters()), **run_hardware}, flush=True)
    for index in range(start, limit):
        if not overfit:
            # FR : Une seed par epoch permet de retrouver l’ordre des lots après une reprise.
            # EN: An epoch-specific seed restores the batch order after a resume.
            train_loader.generator.manual_seed(config["seed"] + index)
        # FR : Attend le GPU avant de mesurer le temps mural de l’epoch.
        # EN: Waits for the GPU before measuring epoch wall time.
        synchronize(device)
        begin = time.perf_counter()
        train_loss = run_epoch(model, fixed_batch if overfit else train_loader, device, optimizer)
        val_loss = None if overfit else run_epoch(model, val_loader, device)
        synchronize(device)
        row = {"step" if overfit else "epoch": index + 1, "train_loss": train_loss,
               "seconds": time.perf_counter() - begin}
        if val_loss is not None:
            row["val_loss"] = val_loss
        history.append(row)
        # FR : Sélectionne le modèle avec la validation ; E0 utilise la loss de son lot.
        # EN: Selects the model using validation; E0 uses its batch loss.
        score = train_loss if overfit else val_loss
        improved = score < best
        best = min(best, score)
        if not overfit or (index + 1) % 50 == 0 or index + 1 == limit:
            saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": config, "completed": index + 1, "best": best, "history": history}
            # FR : last sert à reprendre ; best sert à évaluer le checkpoint retenu.
            # EN: last is for resuming; best is for evaluating the selected checkpoint.
            torch.save(saved, checkpoint_dir / "last.pt")
            if improved or overfit:
                torch.save(saved, checkpoint_dir / "best.pt")
            # FR : Garde les nombres bruts derrière la courbe.
            # EN: Keeps the raw numbers behind the plot.
            with (output / "losses.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
            print(row, flush=True)
    plot_losses(history, output / "loss.png", overfit)
    # FR : Vérifie aussi les mots générés librement sur le lot mémorisé.
    # EN: Also checks freely generated words on the memorized batch.
    if overfit:
        from src.generate import generate
        from src.tokenizer import Tokenizer
        images, _, targets = fixed_batch[0]
        references = [Tokenizer().decode(row[row != -100]) for row in targets]
        predictions = generate(model, images.to(device))
        final_loss = run_epoch(model, fixed_batch, device)
        write_json(output / "results.json", {
            "purpose": "Sanity check on one training batch only", "seed": config["seed"],
            "loss": final_loss, "exact_match": sum(a == b for a, b in zip(predictions, references)) / len(references),
            "references": references, "predictions": predictions, "hardware": run_hardware,
        })


if __name__ == "__main__":
    main()
