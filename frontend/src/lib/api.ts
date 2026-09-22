import type { CommentItem } from "../types"

export interface ApiComment {
    comment_id: string
    video_id: string
    text: string
    author: string
    likes: number
    published_at: string
    is_toxic: boolean
    reasons: string[]
    score: number
}

const API_BASE = import.meta.env.VITE_API_URL ?? ""

async function request<T>(path: string): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`)
    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText} — ${API_BASE}${path}`)
    }
    return res.json() as Promise<T>
}

export async function fetchVideoComments(url: string): Promise<ApiComment[]> {
    return request<ApiComment[]>(`/comments/video?url=${encodeURIComponent(url)}`)
}

export async function fetchSingleComment(
    videoId: string,
    commentId: string,
): Promise<ApiComment> {
    return request<ApiComment>(
        `/comments/comment?video_id=${encodeURIComponent(videoId)}&comment_id=${encodeURIComponent(commentId)}`,
    )
}

export function toCommentItem(api: ApiComment, id: number): CommentItem {
    return {
        id,
        text: api.text,
        prediction: api.is_toxic ? "toxic" : "nontoxic",
        score: api.score,
    }
}
