# RLS Entry Test - Tiny VLM

A small model that looks at an image and writes its description, one letter at a time. For example: `largeredcircle`.

| Experiment | Config | Metric | Result | What it shows |
|---|---|---|---|---|
| Main model | baseline.yaml | Exact match, test | **68.05%**, 1 seed | The model uses the image. |
| E1 blind | blind.yaml | Exact match, test | **1.70%**, 1 seed | Text patterns alone are not enough. |
| E1 blind | blind.yaml | Size / color / shape / relation | **33.28 / 20.10 / 15.30 / 0.00%**, 1 seed | Most scene details are wrong. |
| E0 overfit | overfit.yaml | Fixed-batch loss / exact match | **0.000372 / 100%** | It can learn 16 training examples. |
| S1 CPU | benchmark.yaml, batch 64 | Training images/s | **507.68 ± 24.40**, 5 repeats | Intel i5-10500H, 4 threads. |
| S1 GPU | benchmark.yaml, batch 64 | Training images/s | **4414.42 ± 9.09**, 5 repeats | NVIDIA GTX 1650. |
| Heldout check | baseline.yaml | Exact match, test_heldout | **0.00%**, 1 seed | New color-shape pairs remain difficult. |

Configs are in [configs/](configs/). Accuracy uses one seed, so no standard deviation across seeds is available. The ± values describe timing repeats. Results were recorded on Windows 11 with Python 3.12.14 and PyTorch 2.8.0+cu126.

## How it works

`Image → CNN → 16 visual tokens → Transformer → letters`

The model has 521,307 parameters, 2 decoder blocks and 4 attention heads. Attention is written by hand. It uses no pretrained weights. Comments explain the code in French and English.

## Run it

Use Python 3.12. On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

For CPU only, use `https://download.pytorch.org/whl/cpu` for the PyTorch install, replace `--device cuda` with `--device cpu`, and run the benchmark with `--devices cpu`. On Linux, activate with `source .venv/bin/activate`.

Run these commands from the repository folder:

```powershell
python generate_data.py
python -m pytest -q
python -m src.inspect_batch
python -m src.train --config configs/overfit.yaml --device cuda
python -m src.train --config configs/baseline.yaml --device cuda
python -m src.train --config configs/blind.yaml --device cuda
python -m src.evaluate --checkpoint experiments/main/checkpoints/best.pt --split test --device cuda
python -m src.evaluate --checkpoint experiments/E1_blind/checkpoints/best.pt --split test --device cuda
python -m src.evaluate --checkpoint experiments/main/checkpoints/best.pt --split test_heldout --device cuda
python -m src.analyze_results --predictions experiments/main/results_test_predictions.json
python -m src.analyze_results --predictions experiments/main/results_test_heldout_predictions.json
python -m benchmarks.S1_throughput.benchmark --config configs/benchmark.yaml --devices cpu cuda
python report/build_report.py
```

Keep the official data seed (42) and split sizes. Run S1 after training has finished. It times training on tensors already on the device, with warm-up and CUDA synchronization; data loading and transfers are excluded.

Data and checkpoints stay local. A fresh clone needs to generate the data and train again. To resume training, add `--resume experiments/main/checkpoints/last.pt` to the main training command. The best checkpoint is chosen using validation loss.

## Read the project

| Folder | Contents |
|---|---|
| src/ | Data loading, model, training and evaluation |
| configs/ | Model and training settings |
| tests/ | Attention, shapes, padding and generation checks |
| experiments/ | Results, predictions and loss curves |
| benchmarks/ | Raw timings and CPU/GPU comparison |
| report/ | [Four-page report](report/report.pdf) and its build script |
| docs/ | [Learning notes](docs/learning_notes.md) and [interview questions](docs/interview_questions.md) |

**108 tests pass.** The model works on familiar scenes, but the heldout result shows a clear limit. One seed is not enough to make broad claims.

AI helped with learning, implementation and checks. Details are in [AI_USAGE.md](AI_USAGE.md). The official files and references are listed in [SOURCE.md](SOURCE.md).
