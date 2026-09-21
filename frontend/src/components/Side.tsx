import type { CSSProperties } from "react"

interface SideProps {
    side: "left" | "right"
}

// NOTE: this dark, strong color is tuned for the current light placeholder
// stage background. Once the final dark background for the right-hand
// stage lands, switch this arrow and label color to white.
const ARROW_COLOR = "#111111"

const circleStyle: CSSProperties = {
    width: 76,
    height: 44,
    margin: "0 auto 2px",
    border: "none",
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
}

// One side. Takes the side and builds the arrow and the action label.
export function Side({ side }: SideProps) {
    const isLeft = side === "left"
    return (
        <div style={{ textAlign: "center", width: 140, flexShrink: 0 }}>
            <div style={circleStyle}>
                <span
                    style={{
                        fontSize: 36,
                        fontWeight: 700,
                        color: ARROW_COLOR,
                    }}
                >
                    {isLeft ? "←" : "→"}
                </span>
            </div>
            <p
                style={{
                    margin: 0,
                    fontSize: 18,
                    fontWeight: 700,
                    color: ARROW_COLOR,
                }}
            >
                {isLeft ? "Don't publish" : "Publish"}
            </p>
        </div>
    )
}
