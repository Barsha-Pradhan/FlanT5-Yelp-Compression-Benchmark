"""
yelp_utils.py

Wrapper utilities for Yelp Review Full (5-star) classification.
Deliberately does NOT modify flanutils.py -- mirrors its AG News seq2seq
setup (same prompt-style formatting, same AGNewsSeq2SeqDataset class,
which is generic despite the name) so it plugs straight into the
existing harness in run_sheet4_methods.py / run_all_flant5.py.

Dataset: https://huggingface.co/datasets/Yelp/yelp_review_full
Labels (HF): 0-4  ->  target text: "1 star" ... "5 stars"
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from flanutils import BaseConfig

STAR_LABELS = {0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}


# ==========================================
# Load Yelp Review Full
# ==========================================
def load_yelp_data(cache_dir=None):
    """
    Loads Yelp Review Full via the `datasets` library. Falls back to dummy
    data if unavailable (mirrors flanutils.load_data's dummy-data fallback).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("Yelp/yelp_review_full", cache_dir=cache_dir)
        train_df = ds["train"].to_pandas()
        test_df = ds["test"].to_pandas()
    except Exception as e:
        print(f"⚠️ Could not load Yelp dataset ({e}); creating dummy data")

        def make_dummy(size):
            return pd.DataFrame({
                "label": np.random.choice([0, 1, 2, 3, 4], size),
                "text": ["This is a dummy Yelp review."] * size,
            })
        train_df, test_df = make_dummy(1000), make_dummy(200)

    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("Expected 'text' and 'label' columns in the Yelp dataset")
    return train_df, test_df


# ==========================================
# Prepare Dataset (mirrors prepare_dataset_agnews)
# ==========================================
def prepare_dataset_yelp(train_df, test_df, seed=None):
    seed = seed if seed is not None else BaseConfig.SEED

    def format_dataframe(df):
        data = []
        for _, row in df.iterrows():
            label = row["label"]
            if label not in STAR_LABELS:
                continue
            text_snippet = str(row["text"])[:300]
            prompt = (
                "Classify the following Yelp review into a star rating from "
                "1 star to 5 stars. "
                f"Review: '{text_snippet}'. Predict rating:"
            )
            data.append({
                "input_text": prompt,
                "target_text": STAR_LABELS[label],
                "type": "yelp",
            })
        return pd.DataFrame(data)

    formatted_train = format_dataframe(train_df)
    formatted_test = format_dataframe(test_df)

    train_final, val_final = train_test_split(
        formatted_train, test_size=0.10,
        stratify=formatted_train["target_text"], random_state=seed
    )
    return train_final, val_final, formatted_test


# ==========================================
# Evaluation (5-class counterpart to flanutils.evaluate_seq2seq)
# ==========================================
def evaluate_seq2seq_yelp(model, dataloader, tokenizer, device, model_path=None):
    model.eval()
    preds, gts = [], []
    total_time, total_samples = 0, 0

    # NOTE: was torch.inference_mode() -- quanto's QTensor __torch_function__
    # override tries to set a version_counter on the output tensor during
    # ZeroQuant's generate() call, which inference_mode's stricter tensors
    # don't permit ("Cannot set version_counter for inference tensor").
    # no_grad() has no such restriction and is equally correct for
    # eval-only code (no gradients needed either way).
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            for i in range(input_ids.size(0)):
                start_time = time.time()
                gen_out = model.generate(
                    input_ids=input_ids[i].unsqueeze(0),
                    attention_mask=attention_mask[i].unsqueeze(0),
                    max_new_tokens=5,
                    pad_token_id=tokenizer.pad_token_id,
                )
                total_time += (time.time() - start_time)
                total_samples += 1

                gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True).strip().lower()
                valid_label_tokens = labels[i][labels[i] != -100]
                gt_text = tokenizer.decode(valid_label_tokens, skip_special_tokens=True).strip().lower()

                preds.append(gen_text)
                gts.append(gt_text)

    def to_idx(text):
        # tolerant parse: first digit 1-5 found in the decoded text
        for ch in text:
            if ch in "12345":
                return int(ch) - 1
        return 0  # default to "1 star" on unparseable generation

    y_true = [to_idx(g) for g in gts]
    y_pred = [to_idx(p) for p in preds]

    # Off-by-one accuracy: standard secondary metric for ordinal star-rating
    # tasks, reported IN ADDITION to (never instead of) exact-match accuracy.
    # A prediction one star away from ground truth (e.g. predicting 4 when
    # the true label is 5) counts as correct here. This does not replace
    # "accuracy" -- both are returned so the sheet shows the honest exact
    # number alongside the more forgiving ordinal-tolerant one.
    off_by_one = float(np.mean([abs(t - p) <= 1 for t, p in zip(y_true, y_pred)])) if y_true else 0.0

    model_size = (
        os.path.getsize(model_path) / (1024 ** 2)
        if model_path and os.path.exists(model_path)
        else sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_off_by_one": off_by_one,
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "latency_sec_per_sample": total_time / total_samples if total_samples > 0 else 0,
        "model_size_mb": model_size,
    }