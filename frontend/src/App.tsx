import { startTransition, useCallback, useState } from "react"
import type { CommentItem, Decision, DecisionLogItem } from "./types"
import { EXTRA_SEED, INITIAL_QUEUE } from "./data/seed"
import { extractCommentId, extractVideoId } from "./lib/youtube"
import {
    fetchSingleComment,
    fetchVideoComments,
    toCommentItem,
} from "./lib/api"
import { Sidebar } from "./components/Sidebar"
import { Screen } from "./components/Screen"

// App shell: a fixed left sidebar (home, video link, single-comment lookup,
// load more) plus a main stage on the right showing the swipeable queue.
// App still owns the queue.

// App. The top of everything. Owns the queue and hands it down.
export default function App() {
    const [queue, setQueue] = useState<CommentItem[]>(INITIAL_QUEUE)
    const [, setLog] = useState<DecisionLogItem[]>([])
    const [nextId, setNextId] = useState<number>(INITIAL_QUEUE.length + 1)
    const [currentVideoId, setCurrentVideoId] = useState<string | null>(null)
    const [error, setError] = useState<string | null>(null)

    // Pasted a video link. Loads 10 comments from that video.
    const loadVideo = useCallback(
        async (link: string) => {
            setError(null)
            try {
                const videoId = extractVideoId(link)
                if (!videoId) {
                    setError("Invalid YouTube URL")
                    return
                }
                const comments = await fetchVideoComments(link)
                startTransition(() => {
                    setQueue(
                        comments.map((c, i) => toCommentItem(c, nextId + i)),
                    )
                    setNextId((id) => id + comments.length)
                    setCurrentVideoId(videoId)
                })
            } catch (e) {
                setError(e instanceof Error ? e.message : String(e))
            }
        },
        [nextId],
    )

    // Pasted a comment id or link. Brings only that comment.
    const loadComment = useCallback(
        async (idOrLink: string) => {
            setError(null)
            try {
                const videoId = extractVideoId(idOrLink)
                const commentId = extractCommentId(idOrLink)
                if (!videoId || !commentId) {
                    setError(
                        "Please paste a full YouTube comment URL (must include v= and lc=)",
                    )
                    return
                }
                const comment = await fetchSingleComment(videoId, commentId)
                startTransition(() => {
                    setQueue((prev) => [
                        toCommentItem(comment, nextId),
                        ...prev,
                    ])
                    setNextId((id) => id + 1)
                })
            } catch (e) {
                setError(e instanceof Error ? e.message : String(e))
            }
        },
        [nextId],
    )

    // Home. No routing yet, so it returns to the empty state.
    const home = useCallback(() => {
        startTransition(() => setQueue([]))
    }, [])

    // Sidebar button. Adds more comments.
    const loadMore = useCallback(async () => {
        setError(null)
        if (!currentVideoId) {
            // No video loaded yet — fall back to seed.
            startTransition(() => {
                setQueue((prev) => [
                    ...prev,
                    ...EXTRA_SEED.map((item, index) => ({
                        id: nextId + index,
                        text: item.text,
                        prediction: item.prediction,
                        score: item.score,
                    })),
                ])
                setNextId((id) => id + EXTRA_SEED.length)
            })
            return
        }
        try {
            const link = `https://www.youtube.com/watch?v=${currentVideoId}`
            const comments = await fetchVideoComments(link)
            startTransition(() => {
                setQueue((prev) => [
                    ...prev,
                    ...comments.map((c, i) => toCommentItem(c, nextId + i)),
                ])
                setNextId((id) => id + comments.length)
            })
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e))
        }
    }, [currentVideoId, nextId])

    // Decided what to do with a comment.
    const decide = useCallback((comment: CommentItem, action: Decision) => {
        if (action === "skip") {
            startTransition(() => setQueue((q) => [...q.slice(1), q[0]]))
            return
        }
        startTransition(() => {
            setLog((l) => [...l, { id: comment.id, action }])
            setQueue((q) => q.slice(1))
        })
        // TODO: POST the decision to the backend here.
    }, [])

    return (
        <div
            style={{
                width: "100%",
                height: "100%",
                display: "flex",
                flexDirection: "row",
            }}
        >
            <div
                style={{
                    display: "flex",
                    flexDirection: "column",
                    width: 280,
                    flexShrink: 0,
                }}
            >
                <Sidebar
                    onLoadVideo={loadVideo}
                    onLoadComment={loadComment}
                    onHome={home}
                    onLoadMore={loadMore}
                />
                {error && (
                    <div
                        style={{
                            color: "#e5484d",
                            fontSize: 12,
                            padding: "8px 20px",
                            background: "#0f0f0f",
                            wordBreak: "break-word",
                        }}
                    >
                        {error}
                    </div>
                )}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
                <Screen queue={queue} onDecide={decide} />
            </div>
        </div>
    )
}
