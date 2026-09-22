import { useState } from "react"

interface SidebarProps {
    onLoadVideo: (videoId: string) => Promise<void>
    onLoadComment: (commentId: string) => Promise<void>
    onHome: () => void
    onLoadMore: () => void
}

const SIDEBAR_WIDTH = 280

// The left-hand sidebar. Home button, video-link input, the single-comment
// lookup, and the load-more action pinned to the bottom.
export function Sidebar({
    onLoadVideo,
    onLoadComment,
    onHome,
    onLoadMore,
}: SidebarProps) {
    const [videoId, setVideoId] = useState("")
    const [commentId, setCommentId] = useState("")
    const [open, setOpen] = useState(false)
    const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle")
    const [videoStatus, setVideoStatus] = useState<"idle" | "loading" | "done" | "error">("idle")

    async function submitVideo() {
        if (videoId.trim() === "") return
        setVideoStatus("loading")
        try {
            await onLoadVideo(videoId.trim())
            setVideoStatus("done")
            setVideoId("")
        } catch {
            setVideoStatus("error")
        }
    }

    async function submitComment() {
        if (commentId.trim() === "") return
        setStatus("loading")
        try {
            await onLoadComment(commentId.trim())
            setStatus("done")
            setCommentId("")
        } catch {
            setStatus("error")
        }
    }

    return (
        <div
            style={{
                width: SIDEBAR_WIDTH,
                height: "100%",
                flexShrink: 0,
                background: "#0f0f0f",
                boxSizing: "border-box",
                padding: 20,
                display: "flex",
                flexDirection: "column",
                gap: 20,
            }}
        >
            {/* Home button. Red play button with a white triangle, plus the app name. */}
            <button
                onClick={onHome}
                title="Home"
                style={{
                    border: "none",
                    background: "transparent",
                    padding: 0,
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                }}
            >
                <svg
                    width="36"
                    height="26"
                    viewBox="0 0 36 26"
                    aria-label="Home"
                >
                    <rect width="36" height="26" rx="7" fill="#FF0000" />
                    <path d="M14 7.5 L14 18.5 L24 13 Z" fill="#FFFFFF" />
                </svg>
                <span
                    style={{
                        color: "#ffffff",
                        fontWeight: 900,
                        fontSize: 18,
                        letterSpacing: 0.5,
                    }}
                >
                    Moderator
                </span>
            </button>

            {/* Video link input. Enter loads 10 comments from that video. */}
            <div style={{ position: "relative" }}>
                <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    aria-hidden="true"
                    style={{
                        position: "absolute",
                        top: "50%",
                        left: 14,
                        transform: "translateY(-50%)",
                        pointerEvents: "none",
                    }}
                >
                    <circle
                        cx="7"
                        cy="7"
                        r="5"
                        fill="none"
                        stroke="#8a8a8a"
                        strokeWidth="1.5"
                    />
                    <line
                        x1="11"
                        y1="11"
                        x2="15"
                        y2="15"
                        stroke="#8a8a8a"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                    />
                </svg>
                <input
                    value={videoId}
                    onChange={(e) => setVideoId(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") submitVideo()
                    }}
                    style={{
                        width: "100%",
                        height: 38,
                        border: "none",
                        borderRadius: 19,
                        padding: "0 16px 0 38px",
                        fontSize: 13,
                        background: "#3a3a3a",
                        color: "#ffffff",
                        outline: "none",
                        boxSizing: "border-box",
                    }}
                />
                {videoStatus === "loading" && (
                    <span style={{ fontSize: 12, color: "#9a9a9a" }}>Loading...</span>
                )}
                {videoStatus === "done" && (
                    <span style={{ fontSize: 12, color: "#30a46c" }}>Comments added to queue</span>
                )}
                {videoStatus === "error" && (
                    <span style={{ fontSize: 12, color: "#e5484d" }}>Video not found</span>
                )}
            </div>

            {/* Single-comment lookup. Arrow toggles a field for an id or a link. */}
            <div>
                <button
                    onClick={() => setOpen((o) => !o)}
                    style={{
                        width: "100%",
                        border: "none",
                        background: "transparent",
                        color: "#d0d0d0",
                        cursor: "pointer",
                        fontSize: 16,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "8px 0",
                    }}
                >
                    <span>Bring a single comment</span>
                    <span
                        style={{
                            display: "inline-block",
                            fontSize: 20,
                            color: "#ffffff",
                            transform: open ? "rotate(180deg)" : "none",
                        }}
                    >
                        ▾
                    </span>
                </button>

                {open && (
                    <div
                        style={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 8,
                            marginTop: 4,
                        }}
                    >
                        <input
                            value={commentId}
                            onChange={(e) => setCommentId(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === "Enter") submitComment()
                            }}
                            style={{
                                width: "100%",
                                height: 36,
                                border: "none",
                                borderRadius: 18,
                                padding: "0 14px",
                                fontSize: 13,
                                background: "#3a3a3a",
                                color: "#ffffff",
                                outline: "none",
                                boxSizing: "border-box",
                            }}
                        />
                        <button
                            onClick={submitComment}
                            style={{
                                width: "100%",
                                border: "none",
                                borderRadius: 18,
                                padding: "8px 16px",
                                fontSize: 14,
                                fontWeight: 700,
                                background: "#FF0000",
                                color: "#ffffff",
                                cursor: "pointer",
                            }}
                        >
                            Load
                        </button>
                        {status === "loading" && (
                            <span style={{ color: "#9a9a9a", fontSize: 12 }}>
                                Loading...
                            </span>
                        )}
                        {status === "done" && (
                            <span style={{ color: "#30a46c", fontSize: 12 }}>
                                Comment added to queue
                            </span>
                        )}
                        {status === "error" && (
                            <span style={{ color: "#e5484d", fontSize: 12 }}>
                                Not found
                            </span>
                        )}
                    </div>
                )}
            </div>

            {/* Pushed to the bottom. Adds more comments to the queue. */}
            <button
                onClick={onLoadMore}
                style={{
                    marginTop: "auto",
                    width: "100%",
                    border: "none",
                    borderRadius: 24,
                    padding: "14px 0",
                    fontSize: 16,
                    fontWeight: 700,
                    background: "#FF0000",
                    color: "#ffffff",
                    cursor: "pointer",
                }}
            >
                Load more comments
            </button>
        </div>
    )
}
