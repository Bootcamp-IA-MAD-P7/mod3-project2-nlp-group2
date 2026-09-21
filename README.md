# Toxic Comment Classifier — NLP Group 2

Clasificador multi-label de comentarios tóxicos en inglés. Modelo principal: TF-IDF + Logistic Regression (predice 12 etiquetas). Modelo secundario: DistilBERT fine-tuned sobre youtoxic (predice solo `IsToxic` en binario).

---

## Etiquetas predichas (12)

`IsToxic` · `IsAbusive` · `IsThreat` · `IsProvocative` · `IsObscene` · `IsHatespeech` · `IsRacist` · `IsNationalist` · `IsSexist` · `IsHomophobic` · `IsReligiousHate` · `IsRadicalism`

> Las etiquetas `IsNationalist`, `IsSexist`, `IsHomophobic` e `IsRadicalism` tienen muy pocos ejemplos positivos en el dataset (1000 comentarios), por lo que el modelo predice siempre 0 para ellas.

---

## Estructura del proyecto

```
├── data/
│   ├── raw/                        # Dataset original (youtoxic_english_1000.csv)
│   └── processed/                  # CSVs procesados (train / val / test)
├── models/                         # Modelos entrenados (no se suben a git)
│   ├── tfidf_lr.joblib
│   └── distilbert_youtoxic/
├── notebooks/
│   ├── 01_eda.ipynb                # Análisis exploratorio
│   ├── pipeline_documentation.ipynb
│   └── train_distilbert_colab.ipynb
├── src/
│   ├── data/
│   │   └── prepare_dataset.py      # Limpieza y split del dataset
│   ├── model/
│   │   ├── train.py                # Entrena TF-IDF + LR (multi-label)
│   │   ├── evaluate.py             # Evalúa TF-IDF + LR en test set
│   │   ├── evaluate_distilbert.py  # Evalúa DistilBERT en test set
│   │   └── predict.py              # Inferencia
│   └── app/
│       └── streamlit_app.py        # Interfaz web
├── pyproject.toml
└── requirements.txt
```

---

## Dataset

- Fuente: `youtoxic_english_1000.csv` — 1000 comentarios en inglés etiquetados manualmente
- Split estratificado por `IsToxic`:
  - Train: 700 filas
  - Val: 150 filas
  - Test: 150 filas
- Cada CSV tiene 14 columnas: `CommentId`, `clean_text` + 12 etiquetas (valores 0/1)

---

## Preprocesamiento

`src/data/prepare_dataset.py` aplica sobre el texto original:

1. Lowercase
2. Eliminación de URLs y menciones (`@usuario`)
3. Eliminación de puntuación y números
4. Eliminación de stopwords (NLTK)
5. Lematización (WordNetLemmatizer)

```bash
uv run python src/data/prepare_dataset.py
```

Genera los tres CSVs en `data/processed/`.

---

## Modelos

### TF-IDF + Logistic Regression (baseline)

Desarrollado por Kari. Pipeline personalizado (`MultiLabelPipeline`) que:

- Vectoriza el texto con `TfidfVectorizer` (10.000 features, unigramas + bigramas)
- Entrena un clasificador por etiqueta: `LogisticRegression(class_weight="balanced")` si hay ejemplos positivos, `DummyClassifier(most_frequent)` si no los hay
- Serializado con `joblib`

Disponible en Hugging Face: [KariRomero/tfidf-lr-youtoxic](https://huggingface.co/KariRomero/tfidf-lr-youtoxic)

**Entrenar:**
```bash
uv run python src/model/train.py
```

**Descargar modelo ya entrenado:**
```bash
hf download KariRomero/tfidf-lr-youtoxic tfidf_lr.joblib --local-dir models/
```

**Evaluar en test set:**
```bash
uv run python src/model/evaluate.py
```

---

### DistilBERT fine-tuned

Desarrollado por Kari. Fine-tuning de `distilbert-base-uncased` sobre el dataset youtoxic.

Disponible en Hugging Face: [KariRomero/distilbert-youtoxic](https://huggingface.co/KariRomero/distilbert-youtoxic)

**Entrenar** (requiere GPU — usar Google Colab):
1. Abre `notebooks/train_distilbert_colab.ipynb` en [Google Colab](https://colab.research.google.com)
2. Activa GPU: Entorno de ejecución → Cambiar tipo → T4 GPU
3. Sube `data/processed/train.csv` y `data/processed/val.csv` cuando se pida
4. Ejecuta todas las celdas (~5-10 min)
5. Descarga el modelo y descomprime en `models/distilbert_youtoxic/`

**Descargar modelo ya entrenado:**
```bash
hf download KariRomero/distilbert-youtoxic --local-dir models/distilbert_youtoxic
```

**Evaluar en test set:**
```bash
uv run python src/model/evaluate_distilbert.py
```

---

## Resultados en test set (IsToxic)

| Modelo | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| TF-IDF + LR | 72% | 70% | 70% | 70% |
| DistilBERT | **77%** | **73%** | **78%** | 68% |

---

## Instalación

Requiere Python 3.13.

```bash
# Con uv (recomendado)
uv sync

# Con pip
pip install -r requirements.txt
```

Para NLTK, la primera vez hay que descargar los recursos:
```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

---

## Rama de trabajo

`feature/toxic-filter` — repositorio: [Bootcamp-IA-MAD-P7/mod3-project2-nlp-group2](https://github.com/Bootcamp-IA-MAD-P7/mod3-project2-nlp-group2)
