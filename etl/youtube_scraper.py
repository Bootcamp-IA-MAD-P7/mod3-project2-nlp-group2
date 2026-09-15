import os
import random
import time

import polars as pl
from dotenv import load_dotenv
from datetime import datetime
from googleapiclient.discovery import build

load_dotenv()

API_KEY: str = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

SEARCH_QUERIES: list[str] = [
    "white genocide",
    "great replacement theory",
    "feminism is cancer",
    "men are better than women",
    "islam is evil",
    "christianity is fake",
    "trans women aren't real women",
    "black crime statistics",
    "jews control media",
    "illegal immigrants criminals",
    "fat acceptance is wrong",
    "antifa terrorists",
    "all lives matter protest",
    "groomer agenda schools",
    "replacement migration",
    "white pride",
    "incel revolution",
    "trad wife debate",
    "race realism",
]

VIDEOS_PER_QUERY: int = 40
COMMENTS_PER_VIDEO: int = 30
OUTPUT_PATH: str = (
    f"data/raw/scraped/youtube_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)


def search_videos(query: str, max_results: int = VIDEOS_PER_QUERY) -> list[str]:
    response = (
        youtube.search()
        .list(
            part="id",
            q=query,
            type="video",
            maxResults=max_results,
            relevanceLanguage="en",
        )
        .execute()
    )
    return [item["id"]["videoId"] for item in response.get("items", [])]


def fetch_comments(video_id: str, max_comments: int = COMMENTS_PER_VIDEO) -> list[dict]:
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


def scrape_by_keywords(
    queries: list[str],
    videos_per_query: int = VIDEOS_PER_QUERY,
    comments_per_video: int = COMMENTS_PER_VIDEO,
    shuffle: bool = True,
    output_path: str = OUTPUT_PATH,
) -> pl.DataFrame:

    if shuffle:
        queries = queries.copy()
        random.shuffle(queries)

    all_comments: list[dict] = []
    video_ids_seen: set[str] = set()

    for query in queries:
        print(f"\nSearching: '{query}'")
        try:
            video_ids = search_videos(query, max_results=videos_per_query)
        except Exception as e:  # noqa: BLE001
            print(f"  → Search failed for '{query}': {e}")
            continue

        for video_id in video_ids:
            if video_id in video_ids_seen:
                print(f"  → Skipping duplicate video: {video_id}")
                continue
            video_ids_seen.add(video_id)

            print(f"  Fetching comments for video: {video_id}")
            try:
                comments = fetch_comments(video_id, max_comments=comments_per_video)
                all_comments.extend(comments)
                print(f"  → {len(comments)} comments fetched")
            except Exception as e:  # noqa: BLE001
                print(f"  → Skipping {video_id}: {e}")

            time.sleep(1)

    df = pl.DataFrame(all_comments)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write_csv(output_path)
    print(f"\nSaved {len(df)} total comments to {output_path}")
    return df


if __name__ == "__main__":
    scrape_by_keywords(
        queries=SEARCH_QUERIES,
        shuffle=True,
    )
