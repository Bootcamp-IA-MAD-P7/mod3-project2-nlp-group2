import { useCallback, type CSSProperties } from "react"
import {
    animate,
    motion,
    useMotionValue,
    useTransform,
    type PanInfo,
} from "framer-motion"
import type { CommentItem, Decision } from "../types"

interface CardProps {
    comment: CommentItem
    isTop: boolean
    depth: number
    onDecide: (comment: CommentItem, action: Decision) => void
}

const THRESHOLD = 200 // drag distance in px to dismiss a card

// Button style. Lives out here because it is fixed and does not depend on data.
const buttonStyle: CSSProperties = {
    background: "transparent",
    border: "none",
    borderRadius: 20,
    fontWeight: 700,
    padding: "8px 16px",
    cursor: "pointer",
}

export function Card({ comment, isTop, depth, onDecide }: CardProps) {
    const x = useMotionValue(0)
    const rotate = useTransform(x, [-200, 200], [-18, 18]) // rotates with the drag
    const toxic = comment.prediction === "toxic"
    const verdictLabel = toxic ? "toxic" : "not toxic"

    const handleDragEnd = useCallback(
        (_event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
            const wentRight = info.offset.x > THRESHOLD || info.velocity.x > 800
            const wentLeft =
                info.offset.x < -THRESHOLD || info.velocity.x < -800
            if (wentRight) {
                animate(x, 600, { duration: 0.3 })
                onDecide(comment, "approve")
            } else if (wentLeft) {
                animate(x, -600, { duration: 0.3 })
                onDecide(comment, "reject")
            } else {
                animate(x, 0, { type: "spring", stiffness: 300 }) // back to center
            }
        },
        [comment, onDecide, x]
    )

    return (
        <motion.div
            drag={isTop ? "x" : false}
            onDragEnd={handleDragEnd}
            initial={{ y: -60, opacity: 0 }} // enters from the top
            animate={{ y: depth * 12, scale: 1 - depth * 0.05, opacity: 1 }}
            style={{
                x,
                rotate,
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                borderRadius: 20,
                background: toxic
                    ? "radial-gradient(circle at center, #FCEBEB 5%, #FF0000 150%)"
                    : "radial-gradient(circle at center, #EAF3DE 5%, #074a01 150%)",
                boxShadow: "0 4px 14px rgba(0,0,0,0.15)", // contained shadow, not a large blur
                padding: 28,
                boxSizing: "border-box",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 20,
            }}
        >
            {/* Verdict as plain text near the top edge, no pill. Prediction and score together. */}
            <span
                style={{
                    fontSize: 20,
                    fontWeight: 700,
                    color: toxic ? "#e5484d" : "#30a46c",
                }}
            >
                {verdictLabel} {comment.score.toFixed(2)}
            </span>

            {/* The comment text. Anonymous card: no author, photo, or date. */}
            <p style={{ margin: "auto 10px", fontSize: 36, wordBreak: "break-word", overflowWrap: "break-word", overflow: "hidden" }}>
                {comment.text}
            </p>

            {comment.reasons.length > 0 && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 12 }}>
                    {comment.reasons.map((r) => (
                        <span
                            key={r}
                            style={{
                                background: "#7f1d1d",
                                color: "#fca5a5",
                                borderRadius: 12,
                                padding: "2px 10px",
                                fontSize: 12,
                                fontWeight: 600,
                            }}
                        >
                            {r}
                        </span>
                    ))}
                </div>
            )}

            {isTop && (
                <div
                    style={{
                        position: "absolute",
                        bottom: 20,
                        left: 0,
                        right: 0,
                        display: "flex",
                        justifyContent: "flex-end",
                        paddingRight: 20,
                    }}
                >
                    <button
                        style={buttonStyle}
                        onClick={() => onDecide(comment, "skip")}
                    >
                        ↻ Later
                    </button>
                </div>
            )}
        </motion.div>
    )
}
