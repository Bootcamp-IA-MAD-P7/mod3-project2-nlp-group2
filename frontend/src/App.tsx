import { startTransition, useCallback, useState } from "react"
import type { CommentItem, Decision, DecisionLogItem } from "./types"
import { EXTRA_SEED, INITIAL_QUEUE } from "./data/seed"
import { extractCommentId, extractVideoId } from "./lib/youtube"
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

    // The moderator swiped a card.
    const decide = useCallback((comment: CommentItem, action: Decision) => {
        if (action === "skip") {
            startTransition(() => {
                setQueue((q) => [...q.slice(1), q[0]]) // send card to the end
            })
            return
        }
        startTransition(() => {
            setLog((l) => [...l, { id: comment.id, action }]) // save the decision
            setQueue((q) => q.slice(1))
        })
        // In the real app, POST the decision to the backend here.
    }, [])

    // Sidebar button. Adds more comments.
    const loadMore = useCallback(() => {
        // TODO: real YouTube API call goes here.
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
    }, [nextId])

    // Pasted a video link. Loads 10 comments from that video.
    const loadVideo = useCallback(
        (link: string) => {
            const videoId = extractVideoId(link)
            console.log("Loading comments for video:", videoId)
            // TODO: real YouTube API call with videoId, 10 comments.
            // Without backend, replace the queue with 10 example comments.
            startTransition(() => {
                const fresh: CommentItem[] = Array.from(
                    { length: 10 },
                    (_, i) => {
                        const sample = EXTRA_SEED[i % EXTRA_SEED.length]
                        return {
                            id: nextId + i,
                            text: sample.text,
                            prediction: sample.prediction,
                            score: sample.score,
                        }
                    }
                )
                setQueue(fresh)
                setNextId((id) => id + 10)
            })
        },
        [nextId]
    )

    // Pasted a comment id or link. Brings only that comment.
    const loadComment = useCallback(
        (idOrLink: string) => {
            const commentId = extractCommentId(idOrLink)
            // TODO: real YouTube API call, comments.list by id.
            startTransition(() => {
                const one: CommentItem = {
                    id: nextId,
                    text: `Example comment for id ${commentId}`,
                    prediction: "toxic",
                    score: 0.75,
                }
                setQueue((prev) => [...prev, one])
                setNextId((id) => id + 1)
            })
        },
        [nextId]
    )

    // Home. No routing yet, so it returns to the empty state.
    const home = useCallback(() => {
        startTransition(() => setQueue([]))
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
            <Sidebar
                onLoadVideo={loadVideo}
                onLoadComment={loadComment}
                onHome={home}
                onLoadMore={loadMore}
            />
            <div style={{ flex: 1, minWidth: 0 }}>
                <Screen queue={queue} onDecide={decide} />
            </div>
        </div>
    )
}
