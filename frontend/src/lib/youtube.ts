// Reads the video id from a YouTube link. Returns null if it cannot.
export function extractVideoId(link: string): string | null {
    try {
        const url = new URL(link.trim())
        const v = url.searchParams.get("v")
        if (v) return v
        if (url.hostname.includes("youtu.be")) {
            return url.pathname.slice(1) || null
        }
    } catch {
        // not a valid URL, ignore
    }
    return null
}

// Accepts an id or a comment link. If it is a link with lc=, pulls the id out.
export function extractCommentId(value: string): string {
    const v = value.trim()
    const match = v.match(/[?&]lc=([^&]+)/)
    if (match) return decodeURIComponent(match[1])
    return v
}
