import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from googleapiclient.discovery import build
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

load_dotenv()

API_KEY: str = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

DISTILBERT_MODEL = "Anahia/distilbert-jigsaw-toxicity-multilabel"
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

GATEKEEPER_THRESHOLD = 0.6
ZEROSHOT_ONLY_THRESHOLD = 0.7

thresholds_path = hf_hub_download(
    repo_id="Anahia/distilbert-jigsaw-toxicity-multilabel", filename="thresholds.json"
)

with open(thresholds_path) as f:
    best_thresholds = json.load(f)

HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

LABEL_DISPLAY = {
    "IsToxic": "toxic",
    "IsAbusive": "abusive",
    "IsThreat": "threat",
    "IsProvocative": "provocative",
    "IsObscene": "obscene",
    "IsHatespeech": "hate speech",
    "IsRacist": "racist",
    "IsNationalist": "nationalist",
    "IsSexist": "sexist",
    "IsHomophobic": "homophobic",
    "IsReligiousHate": "religious hate",
    "IsRadicalism": "radicalism",
}

app = FastAPI()

# Allows the Vite dev server (frontend) to call this API from the browser.
if os.getenv("ENVIRONMENT") == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


class CommentResponse(BaseModel):
    comment_id: str
    video_id: str
    text: str
    author: str
    likes: int
    published_at: str
    is_toxic: bool
    reasons: list[str]
    score: float


def get_distilbert_probs(text: str) -> dict[str, float]:
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{DISTILBERT_MODEL}",
        headers=HF_HEADERS,
        json={"inputs": text},
    )
    out = response.json()
    if isinstance(out, list) and isinstance(out[0], list):
        out = out[0]
    probs = {item["label"]: float(item["score"]) for item in out}
    return {label: probs.get(label, 0.0) for label in DISTILBERT_LABELS}


def predict(text: str) -> tuple[dict[str, bool], float]:
    db_probs = get_distilbert_probs(text)

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{ZEROSHOT_MODEL}",
        headers=HF_HEADERS,
        json={
            "inputs": text,
            "parameters": {
                "candidate_labels": list(ZEROSHOT_LABEL_MAP.keys()),
                "multi_label": True,
            },
        },
    )
    zs_out = response.json()
    zs_scores = dict(zip(zs_out["labels"], zs_out["scores"]))

    results = {}

    for label in DISTILBERT_LABELS:
        results[label] = db_probs[label] >= best_thresholds.get(label, 0.5)

    toxic_signal = max(db_probs["IsToxic"], db_probs.get("IsHatespeech", 0))

    for zs_key, col in ZEROSHOT_LABEL_MAP.items():
        if col in results:
            continue
        if toxic_signal < GATEKEEPER_THRESHOLD:
            results[col] = False
        else:
            results[col] = zs_scores.get(zs_key, 0) >= ZEROSHOT_ONLY_THRESHOLD

    return results, toxic_signal


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|/)([\w-]{11})", url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    return match.group(1)


def fetch_comments(video_id: str, max_comments: int = 10) -> list[dict]:
    comments: list[dict] = []
    next_page_token: str | None = None

    while len(comments) < max_comments:
        response = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            )
            .execute()
        )

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "video_id": video_id,
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "text": snippet["textDisplay"],
                    "author": snippet["authorDisplayName"],
                    "likes": snippet["likeCount"],
                    "published_at": snippet["publishedAt"],
                }
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.5)

    return comments


def classify_comment(comment: dict) -> CommentResponse:
    predictions, score = predict(comment["text"])
    is_toxic = any(predictions.values())
    reasons = [
        LABEL_DISPLAY[label]
        for label, is_true in predictions.items()
        if is_true and label != "IsToxic"
    ]
    return CommentResponse(
        comment_id=comment["comment_id"],
        video_id=comment["video_id"],
        text=comment["text"],
        author=comment["author"],
        likes=comment["likes"],
        published_at=comment["published_at"],
        is_toxic=is_toxic,
        reasons=reasons,
        score=score,
    )


@app.get("/comments/video", response_model=list[CommentResponse])
def get_video_comments(url: str):
    video_id = extract_video_id(url)
    comments = fetch_comments(video_id, max_comments=10)
    return [classify_comment(c) for c in comments]


@app.get("/comments/comment", response_model=CommentResponse)
def get_single_comment(video_id: str, comment_id: str):
    comments = fetch_comments(video_id, max_comments=100)
    comment = next((c for c in comments if c["comment_id"] == comment_id), None)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    return classify_comment(comment)


@app.get("/health")
def health():
    return {"status": "ok"}


# Serves the built frontend (Vite) from the container in production.
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount(
        "/", StaticFiles(directory=frontend_dist, html=True), name="frontend"
    )