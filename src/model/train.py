import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
MODEL_PATH = "models/tfidf_lr.joblib"

LABELS = [
    "IsToxic", "IsAbusive", "IsThreat", "IsProvocative", "IsObscene",
    "IsHatespeech", "IsRacist", "IsNationalist", "IsSexist",
    "IsHomophobic", "IsReligiousHate", "IsRadicalism"
]


class MultiLabelPipeline:
    def __init__(self, tfidf, estimators, labels):
        self.tfidf = tfidf
        self.estimators = estimators
        self.labels = labels

    def predict(self, X):
        import numpy as np
        X_tfidf = self.tfidf.transform(X)
        return np.column_stack([e.predict(X_tfidf) for e in self.estimators])


def main():
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)

    X_train, y_train = train["clean_text"], train[LABELS]
    X_val, y_val = val["clean_text"], val[LABELS]

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)

    estimators = []
    for label in LABELS:
        if y_train[label].nunique() < 2:
            estimators.append(DummyClassifier(strategy="most_frequent"))
        else:
            estimators.append(LogisticRegression(max_iter=1000, class_weight="balanced"))
        estimators[-1].fit(X_train_tfidf, y_train[label])

    pipeline = MultiLabelPipeline(tfidf, estimators, LABELS)

    y_pred = pipeline.predict(X_val)
    for i, label in enumerate(LABELS):
        print(f"\n--- {label} ---")
        print(classification_report(y_val.iloc[:, i], y_pred[:, i], zero_division=0))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
