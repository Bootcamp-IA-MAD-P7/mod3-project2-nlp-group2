import type { CommentItem } from "../types"

// --- Seed data. Kept for dev reference only — the live queue starts empty. ---
export const SEED: CommentItem[] = [
    {
        id: 1,
        text: "Great video, thanks for sharing",
        prediction: "nontoxic",
        score: 0.97,
        is_toxic: false,
        reasons: [],
    },
    {
        id: 2,
        text: "You are useless, you should disappear",
        prediction: "toxic",
        score: 0.91,
        is_toxic: true,
        reasons: ["abusive"],
    },
    {
        id: 3,
        text: "I disagree, but I respect your point",
        prediction: "nontoxic",
        score: 0.82,
        is_toxic: false,
        reasons: [],
    },
]

export const EXTRA_SEED_DEV: Array<Omit<CommentItem, "id">> = [
    {
        text: "Excellent editing, everything is clear.",
        prediction: "nontoxic",
        score: 0.95,
        is_toxic: false,
        reasons: [],
    },
    {
        text: "Garbage content, shut the channel down.",
        prediction: "toxic",
        score: 0.88,
        is_toxic: true,
        reasons: ["obscene"],
    },
    {
        text: "This helped me a lot, thanks for the work.",
        prediction: "nontoxic",
        score: 0.9,
        is_toxic: false,
        reasons: [],
    },
]

// Live values. Production starts with no cards until a video is loaded.
export const INITIAL_QUEUE: CommentItem[] = []
export const EXTRA_SEED: CommentItem[] = []
