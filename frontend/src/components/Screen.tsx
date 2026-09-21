import type { CSSProperties } from "react"
import type { CommentItem, Decision } from "../types"
import { Side } from "./Side"
import { ModerationDeck } from "./ModerationDeck"

interface ScreenProps {
    queue: CommentItem[]
    onDecide: (comment: CommentItem, action: Decision) => void
}

const screenStyle: CSSProperties = {
    width: "100%",
    height: "100%",
    minHeight: 560,
    boxSizing: "border-box",
    padding: 24,
    display: "flex",
    flexDirection: "column",
    // Placeholder background for the right-hand stage. Provisional only —
    // the final look for this area is designed separately.
    backgroundColor: "#efefef",
    backgroundImage: "url('/background.png')",
    backgroundSize: "cover",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
    overflow: "hidden",
}

// The screen. Wraps the deck and paints the background. Only displays.
export function Screen({ queue, onDecide }: ScreenProps) {
    return (
        <div
            style={{
                ...screenStyle,
                minHeight: 0,
                padding: "clamp(12px, 2.5vw, 24px)",
            }}
        >
            {/* Header grouped with the deck below, centered together, so it
                sits right above the card instead of floating on its own. */}
            <div
                style={{
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    minHeight: 0,
                    gap: 12,
                }}
            >
                <p
                    style={{
                        margin: 0,
                        fontSize: 22,
                        fontWeight: 900,
                        color: "#5f5e5a",
                        textAlign: "center",
                    }}
                >
                    Swipe the comment to decide
                </p>

                {/* Left, deck, right. */}
                <div
                    style={{
                        width: "min(100%, 1120px)",
                        height: "min(100%, 400px)",
                        display: "grid",
                        gridTemplateColumns:
                            "minmax(90px, 140px) minmax(300px, 720px) minmax(90px, 140px)",
                        alignItems: "center",
                        justifyItems: "center",
                        columnGap: "clamp(10px, 3vw, 48px)",
                    }}
                >
                    <Side side="left" />
                    <div
                        style={{
                            width: "100%",
                            height: "min(100%, 400px)",
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                        }}
                    >
                        <ModerationDeck queue={queue} onDecide={onDecide} />
                    </div>
                    <Side side="right" />
                </div>
            </div>
        </div>
    )
}
