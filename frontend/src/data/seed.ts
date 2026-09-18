import type { CommentItem } from "../types"

// --- Seed data. Test only, while there is no backend. Real comments come from the API. ---
export const SEED: CommentItem[] = [
    {
        id: 1,
        text: "Great video, thanks for sharing",
        prediction: "nontoxic",
        score: 0.97,
    },
    {
        id: 2,
        text: "You are useless, you should disappear",
        prediction: "toxic",
        score: 0.91,
    },
    {
        id: 3,
        text: "I disagree, but I respect your point",
        prediction: "nontoxic",
        score: 0.82,
    },
]

export const EXTRA_SEED: Array<Omit<CommentItem, "id">> = [
    {
        text: "Excellent editing, everything is clear.",
        prediction: "nontoxic",
        score: 0.95,
    },
    {
        text: "Garbage content, shut the channel down.",
        prediction: "toxic",
        score: 0.88,
    },
    {
        text: "This helped me a lot, thanks for the work.",
        prediction: "nontoxic",
        score: 0.9,
    },
]

// Switch. SEED shows cards for the demo. Change to [] to start empty, like production.
export const INITIAL_QUEUE: CommentItem[] = SEED
