export type Prediction = "toxic" | "nontoxic"
export type Decision = "approve" | "reject" | "skip"

export interface CommentItem {
    id: number
    text: string
    prediction: Prediction
    score: number // model confidence for the predicted label, 0 to 1
    is_toxic: boolean
    reasons: string[]
}

export interface DecisionLogItem {
    id: number
    action: Exclude<Decision, "skip">
}
