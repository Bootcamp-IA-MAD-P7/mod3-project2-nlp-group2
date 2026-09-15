import os
from datetime import datetime

import polars as pl
import torch
from transformers import pipeline

BINARY_MODEL = "JungleLee/bert-toxic-comment-classification"
BINARY_THRESHOLD = 0.7

INPUT_PATH = "data/raw/scraped/youtube_comments_20260914_114612.csv"

_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = f"data/processed/labeled_comments_{_RUN_TIMESTAMP}.csv"


def run(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pl.DataFrame:
    device = 0 if torch.cuda.is_available() else -1
    print(
        "GPU detected ✅" if device == 0 else "No GPU ⚠️  — change Colab runtime to T4"
    )

    classifier = pipeline("text-classification", model=BINARY_MODEL, device=device)

    # sanity check — print label strings before the loop
    sample_toxic = classifier("I hate you and I hope you die", truncation=True)[0]
    sample_clean = classifier(
        "This video is really informative thanks", truncation=True
    )[0]
    print(f"Toxic sample  → {sample_toxic}")
    print(f"Clean sample  → {sample_clean}")
    print(f"Using label string: '{sample_toxic['label']}' for toxic detection\n")

    toxic_label = sample_toxic["label"]  # whatever the model actually outputs

    df = pl.read_csv(input_path)
    total = len(df)
    print(f"Loaded {total} comments → {output_path}\n")

    results: list[dict] = []
    skipped_short = 0
    skipped_clean = 0
    failed = 0

    for i, row in enumerate(df.iter_rows(named=True)):
        text = row["text"]

        if not text or len(text.strip()) < 5:
            skipped_short += 1
            continue

        try:
            result = classifier(text, truncation=True, max_length=512)[0]
            is_toxic = (
                result["label"] == toxic_label and result["score"] >= BINARY_THRESHOLD
            )
        except Exception as e:
            print(f"Row {i} failed: {e}")
            failed += 1
            continue

        if not is_toxic:
            skipped_clean += 1
            continue

        results.append(
            {
                "CommentId": row["comment_id"],
                "VideoId": row["video_id"],
                "Text": text,
                "IsToxic": True,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = pl.DataFrame(results)
    out.write_csv(output_path)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Done
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:          {total}
Toxic (kept):   {len(results)}
Clean dropped:  {skipped_clean}
Too short:      {skipped_short}
Errors:         {failed}
Output:         {output_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

    return out


if __name__ == "__main__":
    run()
