import os
from datetime import datetime
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, recall_score

LABELS = ["IsToxic", "IsObscene", "IsThreat", "IsAbusive", "IsHatespeech"]
MODEL_DIR = "models/distilbert_jigsaw"
THRESHOLDS_PATH = "models/distilbert_jigsaw/thresholds.json"
BATCH_SIZE = 32
MAX_LEN = 128

test_df = pd.read_csv("data/processed/test.csv")

with open(THRESHOLDS_PATH) as f:
    thresholds = json.load(f)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


class TestDataset(Dataset):
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


test_dataset = TestDataset(test_df, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

all_probs = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())

test_probs = np.vstack(all_probs)
test_labels = np.vstack(all_labels)

preds = np.zeros_like(test_probs)
for i, label in enumerate(LABELS):
    preds[:, i] = (test_probs[:, i] >= thresholds[label]).astype(int)

per_label_recall = {}
for i, label in enumerate(LABELS):
    per_label_recall[label] = round(
        float(recall_score(test_labels[:, i], preds[:, i], zero_division=0)), 4
    )
    print(f"{label} — Recall: {per_label_recall[label]:.4f}")

print(
    "\n",
    classification_report(test_labels, preds, target_names=LABELS, zero_division=0),
)

os.makedirs("reports", exist_ok=True)

report_dict = classification_report(
    test_labels, preds, target_names=LABELS, zero_division=0, output_dict=True
)

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "split": "test",
    "thresholds_used": thresholds,
    "per_label_recall": per_label_recall,
    "classification_report": report_dict,
}

with open("reports/test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Test results saved to reports/test_results.json")
