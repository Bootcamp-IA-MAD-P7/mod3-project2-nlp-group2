import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/youtoxic_english_1000.csv"
PROCESSED_DIR = "data/processed"

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                  # menciones
    text = re.sub(r"[^a-z\s]", "", text)              # puntuación y números
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def main():
    df = pd.read_csv(RAW_PATH)

    label_cols = [
        "IsToxic", "IsAbusive", "IsThreat", "IsProvocative",
        "IsObscene", "IsHatespeech", "IsRacist", "IsNationalist",
        "IsSexist", "IsHomophobic", "IsReligiousHate", "IsRadicalism",
    ]
    for col in label_cols:
        df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

    df["clean_text"] = df["Text"].astype(str).apply(clean_text)

    df_out = df[["CommentId", "clean_text"] + label_cols]

    train, temp = train_test_split(df_out, test_size=0.30, random_state=42, stratify=df_out["IsToxic"])
    val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["IsToxic"])

    train.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    val.to_csv(f"{PROCESSED_DIR}/val.csv", index=False)
    test.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


if __name__ == "__main__":
    main()
