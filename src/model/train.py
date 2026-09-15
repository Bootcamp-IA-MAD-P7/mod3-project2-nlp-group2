import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
MODEL_PATH = "models/tfidf_lr.joblib"


def main():
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)

    X_train, y_train = train["clean_text"], train["IsToxic"]
    X_val, y_val = val["clean_text"], val["IsToxic"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=["no tóxico", "tóxico"]))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
