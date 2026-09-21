# Toxicity Classification for YouTube Comments

## Overview

A multi-label toxicity classifier fine-tuned on the Jigsaw Toxic Comment dataset, served via a FastAPI backend. It detects 12 toxicity categories in YouTube comments using a DistilBERT model combined with zero-shot classification.

## Toxicity Labels

toxic · obscene · threat · abusive · hate speech · provocative · racist · nationalist · sexist · homophobic · religious hate · radicalism

## Design Philosophy

YouTube comments are overwhelmingly clean — most flagged words appear in neutral or educational contexts. A model optimized for precision would miss real hate speech; one optimized purely for recall would flood moderators with false positives. We optimized for **recall** to catch as much genuine toxicity as possible while keeping thresholds conservative enough to avoid flagging ambiguous language.

Thresholds are per-label and tuned on the validation set via F1 maximization, but the prediction pipeline applies an additional gatekeeper: zero-shot labels only fire when DistilBERT's toxicity signal already exceeds 0.6. This prevents the zero-shot model from flagging politically charged but non-toxic language — words like "nationalist" or "religious" that carry toxicity signal only in genuinely hostile contexts.

## Architecture

### DistilBERT — fine-tuned on Jigsaw

`Anahia/distilbert-jigsaw-toxicity-multilabel` is a DistilBERT model fine-tuned for 5 epochs on the [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (`Arsive/toxicity_classification_jigsaw` on HuggingFace). It covers 5 base labels: toxic, obscene, threat, abusive, hate speech. Thresholds are per-label, optimized via F1 maximization on the Jigsaw validation set. Weighted F1: **0.915**.

### Zero-shot classifier — DeBERTa-v3-small NLI

`cross-encoder/nli-deberta-v3-small` handles the 7 remaining labels: provocative, racist, nationalist, sexist, homophobic, religious hate, radicalism. It is a cross-encoder built on Microsoft's DeBERTa-v3-small architecture, trained on SNLI and MultiNLI. It works by framing each label as a natural language hypothesis and scoring its entailment against the comment text — no task-specific fine-tuning required.

Performance on NLI benchmarks:
- SNLI test accuracy: **91.65%**
- MNLI mismatched accuracy: **87.55%**

### Prediction pipeline (`src/model/distilbert_jigsaw/predict.py`)

For each comment, both models run in parallel:

- The 5 Jigsaw labels use a weighted ensemble: **60% DistilBERT + 40% zero-shot**, with a combined threshold of 0.6.
- The 7 zero-shot-only labels use a two-gate system: DistilBERT's `IsToxic` or `IsHatespeech` score must exceed 0.6 (gatekeeper), and the zero-shot score must independently exceed 0.7. Both gates must pass for a label to fire. This makes the system conservative by design — ambiguous language that doesn't already read as toxic to DistilBERT will not be flagged.

## Alternative Approach — TF-IDF + Logistic Regression & DistilBERT on YouToxic

A parallel implementation trained directly on the YouToxic dataset (1,000 labeled English comments) as an alternative and baseline comparison.

### Models

**TF-IDF + Logistic Regression (multi-label baseline)**
A custom `MultiLabelPipeline` that vectorizes text with TF-IDF (10,000 features, unigrams + bigrams) and trains one `LogisticRegression(class_weight="balanced")` per label — falling back to `DummyClassifier(most_frequent)` for labels with no positive examples. Predicts all 12 categories.

Available on HuggingFace: `KariRomero/tfidf-lr-youtoxic`

**DistilBERT fine-tuned on YouToxic**
`distilbert-base-uncased` fine-tuned on the YouToxic dataset for binary toxicity detection (`IsToxic` only).

Available on HuggingFace: `KariRomero/distilbert-youtoxic`

### Dataset split

Stratified by `IsToxic` — Train: 700 / Val: 150 / Test: 150. Each split has 14 columns: `CommentId`, `clean_text`, and 12 binary labels.

### Preprocessing

Applied in `src/data/prepare_dataset.py`: lowercase, URL and mention removal, punctuation and number removal, stopword removal (NLTK), lemmatization (WordNetLemmatizer).

### Known limitations

`IsNationalist`, `IsSexist`, `IsHomophobic`, and `IsRadicalism` have near-zero positive examples in the 1,000-comment dataset. Both models predict 0 for these labels consistently.

### Setup

```bash
# train TF-IDF + LR
uv run python src/model/train.py

# download pre-trained TF-IDF + LR
hf download KariRomero/tfidf-lr-youtoxic tfidf_lr.joblib --local-dir models/

# evaluate TF-IDF + LR
uv run python src/model/evaluate.py

# download pre-trained DistilBERT
hf download KariRomero/distilbert-youtoxic --local-dir models/distilbert_youtoxic

# evaluate DistilBERT
uv run python src/model/evaluate_distilbert.py
```

DistilBERT fine-tuning requires GPU — use `notebooks/train_distilbert_colab.ipynb` on Google Colab with T4 runtime.

## Jigsaw Validation Results

Evaluated on the Jigsaw validation split. Thresholds were optimized per label via F1 maximization on this same split before evaluation.

| Label | Threshold | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| IsToxic | 0.60 | 0.936 | 0.978 | 0.957 | 3034 |
| IsObscene | 0.50 | 0.897 | 0.940 | 0.918 | 1653 |
| IsThreat | 0.35 | 0.638 | 0.761 | 0.694 | 88 |
| IsAbusive | 0.45 | 0.833 | 0.895 | 0.863 | 1559 |
| IsHatespeech | 0.45 | 0.810 | 0.767 | 0.788 | 262 |
| **weighted avg** | | **0.893** | **0.938** | **0.915** | 6596 |

IsThreat is the weakest label with F1 0.694, expected given its low support (88 examples) and linguistic diversity — threats range from direct language to subtle intimidation. The lower threshold (0.35) reflects this: the model needs less confidence to flag a threat, prioritizing recall over precision, consistent with the project's moderation-first design.

## Test — `test.py`

To validate the labeling pipeline against ground truth, we run `predict.py` on the YouToxic dataset — an independently labeled dataset covering all 12 categories. The script takes only the raw text as input (ignoring the original labels), runs the full prediction pipeline, and saves both the original and predicted labels side by side for direct comparison. A classification report is saved to `src/reports/test_youtoxic.json`.

This gives an honest measure of how well the combined DistilBERT + zero-shot pipeline generalizes to a dataset it was never trained or tuned on.

### YouToxic Results

Tested against 1,000 English comments from the YouToxic dataset, which was not used during training or threshold tuning.

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| IsToxic | 0.74 | 0.71 | 0.72 | 462 |
| IsAbusive | 0.83 | 0.37 | 0.51 | 353 |
| IsThreat | 0.11 | 0.10 | 0.10 | 21 |
| IsProvocative | 0.21 | 0.52 | 0.30 | 161 |
| IsObscene | 0.42 | 0.80 | 0.55 | 100 |
| IsHatespeech | 0.67 | 0.19 | 0.29 | 138 |
| IsRacist | 0.23 | 0.42 | 0.30 | 125 |
| IsNationalist | 0.18 | 0.25 | 0.21 | 8 |
| IsSexist | 0.01 | 1.00 | 0.02 | 1 |
| IsHomophobic | — | — | — | 0 |
| IsReligiousHate | 0.24 | 0.42 | 0.30 | 12 |
| IsRadicalism | — | — | — | 0 |
| **weighted avg** | **0.61** | **0.51** | **0.51** | 1381 |

The Jigsaw-trained labels (IsToxic, IsObscene, IsAbusive) perform strongest, which is expected — DistilBERT was explicitly fine-tuned on those categories. The zero-shot labels (IsRacist, IsNationalist, IsReligiousHate, IsProvocative) show lower precision but reasonable recall, consistent with the conservative gatekeeper design. IsHomophobic and IsRadicalism score zero due to no positive support in this sample, not model failure. IsThreat is the weakest Jigsaw label, likely because threats are rare and linguistically diverse — 21 examples in a 1,000-row sample is too thin for reliable evaluation.

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /comments/video?url={youtube_url}` | 10 comments from a specific video |
| `GET /comments/comment?video_id={id}&comment_id={id}` | Single comment by ID |

All endpoints return classified comments with `is_toxic` and `reasons`.

## Setup

```bash
uv sync
cp .env.example .env   # add YOUTUBE_API_KEY
uv run main.py
```

## Docker

```bash
docker compose up --build
```

The app will be available at `http://localhost:8000`. Models are cached in a Docker volume to avoid re-downloading on restarts. Make sure your `.env` file contains `YOUTUBE_API_KEY`.

## Project Structure

```
├── app/main.py                          # FastAPI backend
├── front/                               # static frontend (HTML/CSS/JS)
├── src/
│   ├── model/distilbert_jigsaw/
│   │   ├── train.py                     # fine-tuning loop
│   │   ├── evaluate.py                  # threshold optimization on val set
│   │   ├── predict.py                   # inference pipeline
│   │   └── test.py                      # validation against YouToxic
│   ├── data/prepare_dataset.py          # Jigsaw dataset prep
│   ├── scraping/                        # YouTube comment scraper
│   └── reports/                         # evaluation and test results
├── models/distilbert_jigsaw/            # saved weights + thresholds.json
├── notebooks/                           # EDA
├── Dockerfile
└── docker-compose.yml
```
