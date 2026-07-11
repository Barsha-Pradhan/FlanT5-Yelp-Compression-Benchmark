# Flan-T5 Yelp Compression Benchmark

A comparative study of six model compression techniques applied to Flan-T5,
evaluated on the [Yelp Review Full](https://huggingface.co/datasets/yelp_review_full)
dataset (5-class sentiment classification, 1–5 stars).

## Models

| Model | Parameters |
|---|---|
| Flan-T5-Small | 80M |
| Flan-T5-Base | 250M |
| Flan-T5-Large | 780M |
| Flan-T5-XL | 3B |

> **Note:** Flan-T5-XL results are not yet included. Full fine-tuning of XL is
> infeasible on the project's hardware (RTX 3060, 12GB VRAM); training-heavy
> methods (Movement Pruning, GAMC) need a LoRA / 8-bit adaptation before XL
> can be benchmarked.

## Compression Methods

| Method | Type |
|---|---|
| [SparseGPT](https://arxiv.org/abs/2301.00774) | One-shot unstructured pruning |
| [HAWQ](https://arxiv.org/abs/1905.03696) | Mixed-precision quantization |
| [ZeroQuant](https://arxiv.org/abs/2206.01861) | Post-training quantization |
| Magnitude Pruning | Weight-magnitude pruning |
| Movement Pruning | Training-aware pruning |
| GAMC | Genetic-algorithm-guided magnitude compression |

## Metrics

- **Accuracy** – exact-match accuracy on the 5-way star rating
- **Accuracy (Off-by-One)** – accuracy allowing predictions within ±1 star
- **Macro-F1** – class-balanced F1 across the 5 rating classes
- **Latency** – average per-sample inference latency (ms)
- **Size** – on-disk model size (MB)

## Results

Exact-match accuracy (%) by method and model size:

| Method    | Small | Base | Large |
|:----------|------:|-----:|------:|
| GAMC      |  31.8 | 39.9 |  53.1 |
| HAWQ      |  46.6 | 50.7 |  56.7 |
| Magnitude |  45.7 | 53.2 |  56.4 |
| Movement  |  44.0 | 50.0 |  51.4 |
| SparseGPT | **47.0** | **53.6** | **59.1** |
| ZeroQuant |  46.7 | 50.5 |  56.8 |

Full per-run numbers (including off-by-one accuracy, Macro-F1, latency, and
size) are in [`results/combined_results.csv`](results/combined_results.csv)
and the individual `results/*_yelp.csv` files.

**Takeaways so far:**
- **SparseGPT is the strongest method at every model size tested**, and the
  margin over the next-best method grows with scale (+2.3 pts at Small,
  +0.4 at Base, +2.3 pts at Large over the runner-up).
- **GAMC lags well behind the other methods at Small and Base** (31.8% and
  39.9%) but closes most of the gap at Large (53.1%), suggesting it needs a
  larger model to have enough redundant capacity to compress well.
- **Movement Pruning is consistently the weakest pruning method**, likely
  because its training-time importance scores were tuned for the original
  task setup rather than Yelp's 5-class schema.
- Accuracy improves with model size for every method, but the amount varies
  a lot — HAWQ only gains 10 points from Small → Large, while GAMC gains
  over 21.

### Accuracy by model size

![Accuracy by model](results/plots/accuracy_by_model.png)

### Accuracy vs. model size (compression trade-off)

![Accuracy vs size](results/plots/accuracy_vs_size.png)

### Accuracy vs. inference latency

![Accuracy vs latency](results/plots/accuracy_vs_latency.png)

### Macro-F1 by model size

![Macro-F1 by model](results/plots/macro_f1_by_model.png)

Plots are regenerated from `results/combined_results.csv` by running:

```bash
python results/generate_plots.py
```

## Dataset

[Yelp Review Full](https://huggingface.co/datasets/yelp_review_full) from
HuggingFace Datasets — 650k train / 50k test reviews labeled 1–5 stars.

## Project Structure

```
code/
├── run_all_flant5.py     # main benchmarking driver across models/methods
├── yelp_utils.py          # Yelp dataset loading and preprocessing
└── flanutils.py           # Flan-T5 loading, generation, evaluation helpers

results/
├── *_yelp.csv              # per-method raw results
├── *_yelp_sheet4_format.xlsx
├── combined_results.csv    # all methods/models combined for analysis
├── generate_plots.py       # regenerates the charts above
└── plots/                  # generated comparison charts

checkpoints/
└── Model checkpoints (not included — too large for git)
```

## Setup

```bash
pip install -r requirements.txt
```

## Status

Small, Base, and Large model sizes are complete for all six methods. XL is
pending a LoRA/8-bit adaptation of the training-heavy methods to fit on
available GPU memory.
