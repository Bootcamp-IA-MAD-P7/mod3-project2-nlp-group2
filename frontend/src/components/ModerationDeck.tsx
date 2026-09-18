import type { CommentItem, Decision } from "../types"
import { Card } from "./Card"

interface ModerationDeckProps {
    queue: CommentItem[]
    onDecide: (comment: CommentItem, action: Decision) => void
}

export function ModerationDeck({ queue, onDecide }: ModerationDeckProps) {
    return (
        <div
            style={{
                position: "relative",
                width: "min(660px, 100%)",
                height: "min(400px, 100%)",
                aspectRatio: "660 / 400",
                margin: "0 auto",
            }}
        >
            {queue
                .slice(0, 3)
                .reverse()
                .map((comment, i, arr) => {
                    const depth = arr.length - 1 - i // 0 is the front card
                    return (
                        <Card
                            key={comment.id}
                            comment={comment}
                            isTop={depth === 0}
                            depth={depth}
                            onDecide={onDecide}
                        />
                    )
                })}
        </div>
    )
}
