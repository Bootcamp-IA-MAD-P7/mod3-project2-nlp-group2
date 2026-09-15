import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/tfidf_lr.joblib"


def main():
    test = pd.read_csv(TEST_PATH)
    X_test, y_test = test["clean_text"], test["IsToxic"]

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)

    print("=== Resultados en Test Set ===\n")
    print(classification_report(y_test, y_pred, target_names=["no tóxico", "tóxico"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(f"  Verdaderos negativos:  {cm[0][0]}  |  Falsos positivos: {cm[0][1]}")
    print(f"  Falsos negativos:      {cm[1][0]}  |  Verdaderos positivos: {cm[1][1]}")


if __name__ == "__main__":
    main()
