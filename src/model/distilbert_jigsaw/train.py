import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import os

LABELS = ["IsToxic", "IsObscene", "IsThreat", "IsAbusive", "IsHatespeech"]
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/distilbert_jigsaw"
EPOCHS = 5
BATCH_SIZE = 32
MAX_LEN = 128
LR = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


class JigsawDataset(Dataset):
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


train_dataset = JigsawDataset(train_df, tokenizer, MAX_LEN)
val_dataset = JigsawDataset(val_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), problem_type="multi_label_classification"
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

loss_fn = torch.nn.BCEWithLogitsLoss()

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(
        f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Best model saved at epoch {epoch + 1}")
