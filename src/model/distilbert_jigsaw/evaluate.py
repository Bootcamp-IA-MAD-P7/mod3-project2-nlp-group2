import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import f1_score, classification_report

LABELS = ["IsToxic", "IsObscene", "IsThreat", "IsAbusive", "IsHatespeech"]
MODEL_DIR = "models/distilbert_jigsaw"
BATCH_SIZE = 32
MAX_LEN = 128

val_df = pd.read_csv("data/processed/val.csv")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


class EvalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.labels = df[LABELS].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]),
        }


val_dataset = EvalDataset(val_df, tokenizer, MAX_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

all_probs = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())

val_probs = np.vstack(all_probs)
val_labels = np.vstack(all_labels)

thresholds = np.arange(0.1, 0.9, 0.05)
best_thresholds = {}

for i, label in enumerate(LABELS):
    scores = [
        f1_score(val_labels[:, i], (val_probs[:, i] >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_t = thresholds[np.argmax(scores)]
    best_thresholds[label] = round(float(best_t), 2)
    print(f"{label} — best threshold: {best_t:.2f} | F1: {max(scores):.4f}")

print("\nBest thresholds:", best_thresholds)

preds = np.zeros_like(val_probs)
for i, label in enumerate(LABELS):
    preds[:, i] = (val_probs[:, i] >= best_thresholds[label]).astype(int)

print(
    "\n", classification_report(val_labels, preds, target_names=LABELS, zero_division=0)
)

with open("models/distilbert_jigsaw/thresholds.json", "w") as f:
    json.dump(best_thresholds, f, indent=2)
print("Thresholds saved.")
