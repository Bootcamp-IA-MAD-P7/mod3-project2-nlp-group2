import torch
import json
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    pipeline,
)

MODEL_DIR = "Anahia/distilbert-jigsaw-toxicity-multilabel"
ZEROSHOT_MODEL = "cross-encoder/nli-deberta-v3-small"

DISTILBERT_LABELS = ["IsToxic", "IsObscene", "IsThreat", "IsAbusive", "IsHatespeech"]

ZEROSHOT_LABEL_MAP = {
    "provocative or inflammatory": "IsProvocative",
    "racist": "IsRacist",
    "nationalist": "IsNationalist",
    "sexist": "IsSexist",
    "homophobic": "IsHomophobic",
    "religious hate": "IsReligiousHate",
    "radicalism or extremism": "IsRadicalism",
}

SHARED_DISTILBERT_WEIGHT = 0.6
SHARED_ZEROSHOT_WEIGHT = 0.4
SHARED_THRESHOLD = 0.6

GATEKEEPER_THRESHOLD = 0.6
ZEROSHOT_ONLY_THRESHOLD = 0.7


thresholds_path = hf_hub_download(
    repo_id="Anahia/distilbert-jigsaw-toxicity-multilabel", filename="thresholds.json"
)

with open(thresholds_path) as f:
    best_thresholds = json.load(f)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
distilbert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilbert_model.to(device)

zero_shot = pipeline(
    "zero-shot-classification",
    model=ZEROSHOT_MODEL,
    device=0 if torch.cuda.is_available() else -1,
)


def get_distilbert_probs(text):
    encoding = tokenizer(
        text, truncation=True, max_length=128, padding="max_length", return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        logits = distilbert_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {label: float(probs[i]) for i, label in enumerate(DISTILBERT_LABELS)}


def predict(text):
    db_probs = get_distilbert_probs(text)

    zs_out = zero_shot(
        text, candidate_labels=list(ZEROSHOT_LABEL_MAP.keys()), multi_label=True
    )
    zs_scores = dict(zip(zs_out["labels"], zs_out["scores"]))

    results = {}

    for label in DISTILBERT_LABELS:
        zs_key = next((k for k, v in ZEROSHOT_LABEL_MAP.items() if v == label), None)
        if zs_key:
            combined = SHARED_DISTILBERT_WEIGHT * db_probs[
                label
            ] + SHARED_ZEROSHOT_WEIGHT * zs_scores.get(zs_key, 0)
            results[label] = combined >= SHARED_THRESHOLD
        else:
            results[label] = db_probs[label] >= best_thresholds.get(label, 0.5)

    toxic_signal = max(db_probs["IsToxic"], db_probs.get("IsHatespeech", 0))

    for zs_key, col in ZEROSHOT_LABEL_MAP.items():
        if col in results:
            continue
        if toxic_signal < GATEKEEPER_THRESHOLD:
            results[col] = False
        else:
            results[col] = zs_scores.get(zs_key, 0) >= ZEROSHOT_ONLY_THRESHOLD

    return results


if __name__ == "__main__":
    test_comments = [
        "I hate all people from that country, they should be removed.",
        "Great video, thanks for sharing!",
        "You are a disgusting pig and I hope you suffer.",
    ]
    for comment in test_comments:
        print(f"\nComment: {comment}")
        print(predict(comment))
