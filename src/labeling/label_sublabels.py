import os
import glob
import polars as pl
import torch
from transformers import pipeline
from google.colab import files

ZEROSHOT_MODEL = "facebook/bart-large-mnli"
ZEROSHOT_THRESHOLD = 0.5
OUTPUT_PATH = "/content/classified_comments.csv"

LABEL_MAP = {
    "abusive": "IsAbusive",
    "threat": "IsThreat",
    "provocative": "IsProvocative",
    "obscene": "IsObscene",
    "hate speech": "IsHatespeech",
    "racist": "IsRacist",
    "nationalist": "IsNationalist",
    "sexist": "IsSexist",
    "homophobic": "IsHomophobic",
    "religious hate": "IsReligiousHate",
    "radicalism": "IsRadicalism",
}

device = 0 if torch.cuda.is_available() else -1
print("GPU detected ✅" if device == 0 else "No GPU ⚠️")

classifier = pipeline("zero-shot-classification", model=ZEROSHOT_MODEL, device=device)

csv_files = glob.glob("/content/labeled_comments_*.csv")
print("Found:", csv_files)
INPUT_PATH = csv_files[0]

df = pl.read_csv(INPUT_PATH)
total = len(df)
print(f"\nLoaded {total} toxic comments from {INPUT_PATH}\n")

results = []
failed = 0

for i, row in enumerate(df.iter_rows(named=True)):
    text = row["Text"]

    try:
        output = classifier(
            text, candidate_labels=list(LABEL_MAP.keys()), multi_label=True
        )
        scores = dict(zip(output["labels"], output["scores"]))
        label_flags = {
            col: scores.get(label, 0) >= ZEROSHOT_THRESHOLD
            for label, col in LABEL_MAP.items()
        }
    except Exception as e:
        print(f"Row {i} failed: {e}")
        failed += 1
        continue

    results.append(
        {
            "CommentId": row["CommentId"],
            "VideoId": row["VideoId"],
            "Text": text,
            "IsToxic": True,
            **label_flags,
        }
    )

    if (i + 1) % 50 == 0 or (i + 1) == total:
        print(f"[{i + 1}/{total}] done | failed: {failed}")

new_data = pl.DataFrame(results)

if os.path.exists(OUTPUT_PATH):
    existing = pl.read_csv(OUTPUT_PATH)
    combined = pl.concat([existing, new_data]).unique(subset=["CommentId"])
    combined.write_csv(OUTPUT_PATH)
    print(f"\nAppended — total rows: {len(combined)}")
else:
    new_data.write_csv(OUTPUT_PATH)
    print(f"\nCreated — {len(new_data)} rows")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Done
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Processed:      {total}
Sub-labeled:    {len(results)}
Errors:         {failed}
Output:         {OUTPUT_PATH}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

files.download(OUTPUT_PATH)
