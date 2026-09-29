# S1 - Training speed

```bash
python -m benchmarks.S1_throughput.benchmark --config configs/benchmark.yaml --devices cpu cuda
```

- Batch sizes: 1, 16, 64 and 256.
- Five warm-up steps, then five repeats of ten training steps.
- Float32, 16 visual tokens and 45 input letters.
- Includes forward, loss, backward, gradient clipping and AdamW.
- Excludes data loading, CPU/GPU transfers and file writes.

CUDA is synchronized before and after each timed repeat. Run this benchmark without another training job.

`raw.csv` holds the timings, `results.json` the mean and sample standard deviation, `hardware.txt` the machine details, and `throughput.png` the plot.
