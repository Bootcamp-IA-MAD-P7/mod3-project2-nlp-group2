import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/distilbert_youtoxic"


def main():
    test = pd.read_csv(TEST_PATH).dropna(subset=["clean_text"])
    texts = test["clean_text"].tolist()
    y_test = test["IsToxic"].tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    y_pred = np.argmax(logits.cpu().numpy(), axis=1)

    print("=== Resultados DistilBERT en Test Set ===\n")
    print(classification_report(y_test, y_pred, target_names=["no tóxico", "tóxico"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(f"  Verdaderos negativos:  {cm[0][0]}  |  Falsos positivos: {cm[0][1]}")
    print(f"  Falsos negativos:      {cm[1][0]}  |  Verdaderos positivos: {cm[1][1]}")


if __name__ == "__main__":
    main()
