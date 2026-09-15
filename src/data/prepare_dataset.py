from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("data/processed", exist_ok=True)

dataset = load_dataset("Arsive/toxicity_classification_jigsaw", cache_dir="data/raw")
train_raw = pd.DataFrame(dataset["train"])
val_raw = pd.DataFrame(dataset["validation"])


def map_columns(df):
    mapped = pd.DataFrame()
    mapped["text"] = df["comment_text"]
    mapped["IsToxic"] = ((df["toxic"] == 1) | (df["severe_toxic"] == 1)).astype(int)
    mapped["IsObscene"] = df["obscene"].astype(int)
    mapped["IsThreat"] = df["threat"].astype(int)
    mapped["IsAbusive"] = df["insult"].astype(int)
    mapped["IsHatespeech"] = df["identity_hate"].astype(int)
    return mapped


train_df = map_columns(train_raw)
val_df = map_columns(val_raw)

train_split, test_split = train_test_split(train_df, test_size=0.15, random_state=42)

train_split.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_split.to_csv("data/processed/test.csv", index=False)

print(f"Train: {len(train_split)} | Val: {len(val_df)} | Test: {len(test_split)}")
