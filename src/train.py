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
    model.train(optimizer is not None)
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.set_grad_enabled(optimizer is not None):
        for images, inputs, targets in loader:
            images, inputs, targets = [x.to(device) for x in (images, inputs, targets)]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, inputs)
            loss = criterion(logits.reshape(-1, 27), targets.reshape(-1))
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = (targets != -100).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
    return total_loss / total_tokens


def main():
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
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        for key in config:
            if key not in ("epochs", "overfit_steps") and config[key] != saved["config"][key]:
                raise ValueError(f"Resume config differs at {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, start = saved["history"], saved["best"], saved["completed"]
    train_loader = make_loader(config, "train", shuffle=True)
    fixed_batch = [next(iter(train_loader))] if overfit else None
    val_loader = None if overfit else make_loader(config, "val")
    limit = config["overfit_steps"] if overfit else config["epochs"]
    run_hardware = hardware(device)
    write_json(output / "config.json", config)
    write_json(output / "hardware.json", run_hardware)
    print({"parameters": sum(p.numel() for p in model.parameters()), **run_hardware}, flush=True)
    for index in range(start, limit):
        if not overfit:
            train_loader.generator.manual_seed(config["seed"] + index)
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
        score = train_loss if overfit else val_loss
        improved = score < best
        best = min(best, score)
        if not overfit or (index + 1) % 50 == 0 or index + 1 == limit:
            saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": config, "completed": index + 1, "best": best, "history": history}
            torch.save(saved, checkpoint_dir / "last.pt")
            if improved or overfit:
                torch.save(saved, checkpoint_dir / "best.pt")
            with (output / "losses.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
            print(row, flush=True)
    plot_losses(history, output / "loss.png", overfit)
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
