# FR : Chronomètre un pas d’entraînement avec des tenseurs déjà sur le périphérique.
# EN: Times a training step with tensors already on the device.

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from src.model.vlm import TinyVLM
from src.utils import hardware, read_config, setup, synchronize, write_json


def measure(config, device, batch_size):
    setup(config["seed"], config["threads"])
    model = TinyVLM(**config["model"]).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # FR : Crée les données avant le timer : chargement et transferts sont exclus.
    # EN: Creates data before timing: loading and transfers are excluded.
    images = torch.rand(batch_size, 3, 64, 64, device=device)
    inputs = torch.randint(0, 27, (batch_size, config["letters"]), device=device)
    targets = torch.randint(0, 27, (batch_size, config["letters"] + 1), device=device)
    criterion = nn.CrossEntropyLoss()

    # FR : Un pas complet inclut forward, loss, backward, clipping et AdamW.
    # EN: A full step includes forward, loss, backward, clipping and AdamW.
    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, inputs)
        loss = criterion(logits.reshape(-1, 27), targets.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # FR : Chauffe le modèle et initialise les états d’AdamW avant les mesures.
    # EN: Warms up the model and initializes AdamW state before measurements.
    for _ in range(config["warmup"]):
        step()
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for repeat in range(config["repeats"]):
        synchronize(device)
        # FR : La synchronisation juste avant et après le travail attend la fin des kernels.
        # EN: Synchronization before and after the work waits for kernels to finish.
        start = time.perf_counter()
        for _ in range(config["steps"]):
            step()
        synchronize(device)
        seconds = time.perf_counter() - start
        # FR : Débit = nombre d’images traitées / durée en secondes.
        # EN: Throughput = number of processed images / elapsed seconds.
        rows.append({"device": device, "batch_size": batch_size, "repeat": repeat + 1,
                     "steps": config["steps"], "seconds": seconds,
                     "images_per_second": batch_size * config["steps"] / seconds,
                     "milliseconds_per_step": seconds * 1000 / config["steps"],
                     "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu"])
    parser.add_argument("--output", default="benchmarks/S1_throughput")
    args = parser.parse_args()
    config = read_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, machines, summary = [], {}, []
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        setup(config["seed"], config["threads"])
        machines[device] = hardware(device)
        for batch_size in config["batch_sizes"]:
            measured = measure(config, device, batch_size)
            rows.extend(measured)
            rates = [row["images_per_second"] for row in measured]
            # FR : L’écart-type décrit les répétitions de timing, pas plusieurs entraînements.
            # EN: Standard deviation describes timing repeats, not multiple training runs.
            result = {"device": device, "batch_size": batch_size, "mean_images_per_second": statistics.mean(rates),
                      "std_images_per_second": statistics.stdev(rates), "n_repeats": len(rates)}
            summary.append(result)
            print(result, flush=True)
            # FR : Sauvegarde les temps bruts pour vérifier les moyennes.
            # EN: Saves raw timings so the averages can be checked.
            with (output / "raw.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    write_json(output / "results.json", {"config": config, "summary": summary, "hardware": machines,
               "scope": "Synthetic resident tensors; forward, loss, backward, gradient clipping and AdamW. No data loading or transfers."})
    (output / "hardware.txt").write_text(str(machines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3.5))
    # FR : Les barres d’erreur montrent l’écart-type des débits.
    # EN: Error bars show the standard deviation of throughput.
    for device in args.devices:
        values = [row for row in summary if row["device"] == device]
        plt.errorbar([r["batch_size"] for r in values], [r["mean_images_per_second"] for r in values],
                     yerr=[r["std_images_per_second"] for r in values], marker="o", capsize=4, label="GPU" if device == "cuda" else "CPU")
    plt.xscale("log", base=2)
    plt.xticks(config["batch_sizes"], [str(size) for size in config["batch_sizes"]])
    plt.xlabel("Batch size")
    plt.ylabel("Training speed (images/s)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output / "throughput.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
