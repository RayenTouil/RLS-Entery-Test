# FR : Petites fonctions communes : seeds, fichiers, matériel et courbes.
# EN: Shared helpers for seeds, files, hardware and plots.

import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# FR : Fixe les sources aléatoires ; les résultats peuvent encore varier entre matériels.
# EN: Sets random seeds; results can still vary across hardware.
def setup(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# FR : Lit les réglages YAML sans les mélanger au code du modèle.
# EN: Reads YAML settings separately from the model code.
def read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# FR : Enregistre les mesures dans un format lisible et réutilisable.
# EN: Saves measurements in a readable, reusable format.
def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# FR : Associe chaque mesure au CPU, GPU et aux versions utilisés.
# EN: Records the CPU, GPU and versions used for each measurement.
def hardware(device):
    cpu = platform.processor()
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {"cpu": cpu, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
            "os": platform.platform(), "threads": torch.get_num_threads(), "cuda": torch.version.cuda}


# FR : CUDA travaille de façon asynchrone : il faut attendre avant de lire le timer.
# EN: CUDA runs asynchronously: wait for it before reading the timer.
def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


# FR : Trace les données enregistrées ; aucune valeur de loss n’est modifiée.
# EN: Plots recorded data without changing any loss values.
def plot_losses(history, path, overfit=False):
    import matplotlib
    # FR : Utilise un rendu sans fenêtre pour les scripts et les notebooks.
    # EN: Uses a non-interactive backend for scripts and notebooks.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axis = "step" if overfit else "epoch"
    plt.figure(figsize=(6, 3.5))
    plt.plot([r[axis] for r in history], [r["train_loss"] for r in history], label="train")
    if not overfit:
        plt.plot([r[axis] for r in history], [r["val_loss"] for r in history], label="validation")
    plt.xlabel(axis)
    plt.ylabel("Cross-entropy (nats / target token)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
