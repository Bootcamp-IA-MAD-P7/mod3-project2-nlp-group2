import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from train import MultiLabelPipeline  # noqa: F401 needed for joblib.load

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/tfidf_lr.joblib"

LABELS = [
    "IsToxic", "IsAbusive", "IsThreat", "IsProvocative", "IsObscene",
    "IsHatespeech", "IsRacist", "IsNationalist", "IsSexist",
    "IsHomophobic", "IsReligiousHate", "IsRadicalism"
]


def main():
    test = pd.read_csv(TEST_PATH)
    X_test, y_test = test["clean_text"], test[LABELS]

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)

    print("=== Resultados en Test Set ===\n")
    for i, label in enumerate(LABELS):
        print(f"\n--- {label} ---")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division=0))


if __name__ == "__main__":
    main()
