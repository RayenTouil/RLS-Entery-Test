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
    images = torch.rand(batch_size, 3, 64, 64, device=device)
    inputs = torch.randint(0, 27, (batch_size, config["letters"]), device=device)
    targets = torch.randint(0, 27, (batch_size, config["letters"] + 1), device=device)
    criterion = nn.CrossEntropyLoss()

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, inputs)
        loss = criterion(logits.reshape(-1, 27), targets.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(config["warmup"]):
        step()
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for repeat in range(config["repeats"]):
        synchronize(device)
        start = time.perf_counter()
        for _ in range(config["steps"]):
            step()
        synchronize(device)
        seconds = time.perf_counter() - start
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
            result = {"device": device, "batch_size": batch_size, "mean_images_per_second": statistics.mean(rates),
                      "std_images_per_second": statistics.stdev(rates), "n_repeats": len(rates)}
            summary.append(result)
            print(result, flush=True)
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
    for device in args.devices:
        values = [row for row in summary if row["device"] == device]
        plt.errorbar([r["batch_size"] for r in values], [r["mean_images_per_second"] for r in values],
                     yerr=[r["std_images_per_second"] for r in values], marker="o", capsize=4, label=device)
    plt.xscale("log", base=2)
    plt.xlabel("Batch size")
    plt.ylabel("Training images / second (mean +/- sample std, 5 repeats)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output / "throughput.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
