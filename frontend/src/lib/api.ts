import type { CommentItem } from "../types"

export interface CommentResponse {
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

export async function fetchVideoComments(videoId: string): Promise<CommentResponse[]> {
    const res = await fetch(`/comments/video?video_id=${encodeURIComponent(videoId)}`)
    if (!res.ok) throw new Error(await res.text())
    return res.json()
}

export async function fetchSingleComment(videoId: string, commentId: string): Promise<CommentResponse> {
    const res = await fetch(`/comments/comment?video_id=${encodeURIComponent(videoId)}&comment_id=${encodeURIComponent(commentId)}`)
    if (!res.ok) throw new Error(await res.text())
    return res.json()
}

export function toCommentItem(api: CommentResponse, id: number): CommentItem {
    return {
        id,
        text: api.text,
        prediction: api.is_toxic ? "toxic" : "nontoxic",
        score: api.score,
    }
}
