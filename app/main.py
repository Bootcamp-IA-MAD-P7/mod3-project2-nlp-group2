import os
import re
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from googleapiclient.discovery import build
from pydantic import BaseModel

from src.model.distilbert_jigsaw.predict import predict

load_dotenv()

API_KEY: str = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

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


class CommentResponse(BaseModel):
    comment_id: str
    video_id: str
    text: str
    author: str
    likes: int
    published_at: str
    is_toxic: bool
    reasons: list[str]


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
    predictions = predict(comment["text"])
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
