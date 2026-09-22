import json
import os

from huggingface_hub import InferenceClient, hf_hub_download

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

client = InferenceClient(token=os.getenv("HF_TOKEN"))


def get_distilbert_probs(text):
    out = client.text_classification(text, model=MODEL_DIR)
    probs = {item["label"]: float(item["score"]) for item in out}
    return {label: probs.get(label, 0.0) for label in DISTILBERT_LABELS}


def predict(text):
    db_probs = get_distilbert_probs(text)

    zs_out = client.zero_shot_classification(
        text,
        labels=list(ZEROSHOT_LABEL_MAP.keys()),
        multi_label=True,
        model=ZEROSHOT_MODEL,
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