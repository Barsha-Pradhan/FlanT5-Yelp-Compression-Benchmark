# Flan-T5 Yelp Compression Benchmark

Benchmarking different compression techniques on Flan-T5 models using the Yelp Review Full dataset.

## Models

- Flan-T5-Small
- Flan-T5-Base
- Flan-T5-Large
- Flan-T5-XL

## Compression Methods

- SparseGPT
- HAWQ
- ZeroQuant
- Magnitude Pruning
- Movement Pruning
- GAMC

## Metrics

- Accuracy
- Macro-F1 Score
- Latency
- Model Size

## Dataset

Yelp Review Full Dataset from HuggingFace.

## Project Structure
code/
├── run_all_flant5.py
├── yelp_utils.py
└── flanutils.py

results/
└── Benchmark results

checkpoints/
└── Model checkpoints (not included)
