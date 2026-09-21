# Explicación completa del proyecto — Toxic Comment Classifier

Este documento explica **qué hace cada parte del proyecto, por qué se hizo así y cómo encaja todo**.

---

## 1. El problema

Dado un comentario en inglés, queremos saber si es tóxico y de qué tipo. No es una clasificación simple de sí/no — un comentario puede ser a la vez abusivo, obsceno y racista. Por eso es un problema **multi-label**: cada comentario puede tener varias etiquetas activas al mismo tiempo.

Las 12 etiquetas que predecimos son:

| Etiqueta | Qué detecta |
|---|---|
| `IsToxic` | Toxicidad general |
| `IsAbusive` | Lenguaje abusivo |
| `IsThreat` | Amenazas |
| `IsProvocative` | Provocación |
| `IsObscene` | Lenguaje obsceno |
| `IsHatespeech` | Discurso de odio |
| `IsRacist` | Racismo |
| `IsNationalist` | Nacionalismo extremo |
| `IsSexist` | Sexismo |
| `IsHomophobic` | Homofobia |
| `IsReligiousHate` | Odio religioso |
| `IsRadicalism` | Radicalismo |

---

## 2. El dataset

**Archivo:** `data/raw/youtoxic_english_1000.csv`

- 1000 comentarios en inglés etiquetados manualmente
- Columna de texto: `Text`
- 12 columnas de etiquetas con valores `True`/`False`
- Problema de **desbalance severo**: la mayoría de comentarios no son tóxicos. Además, etiquetas como `IsNationalist`, `IsSexist`, `IsHomophobic` e `IsRadicalism` tienen tan pocos ejemplos positivos que los modelos no pueden aprenderlas con solo 1000 filas.

---

## 3. Preprocesamiento — `src/data/prepare_dataset.py`

Este script transforma el dataset crudo en tres CSVs listos para entrenar.

### ¿Qué hace?

**Paso 1 — Limpieza del texto** (función `clean_text`):
```
"Check out http://spam.com @user!! You're STUPID 123"
→ "check stupid"
```

1. Convierte a minúsculas
2. Elimina URLs (`http://...`, `www...`)
3. Elimina menciones (`@usuario`)
4. Elimina puntuación y números (deja solo letras y espacios)
5. Elimina stopwords — palabras muy frecuentes sin significado semántico (`the`, `is`, `a`, `you`...)
6. Lematiza — reduce cada palabra a su forma base (`running` → `run`, `better` → `good`)

El resultado se guarda en la columna `clean_text`.

**Paso 2 — Conversión de etiquetas:**
Los valores `True`/`False` del CSV original se convierten a `1`/`0`.

**Paso 3 — Split estratificado:**
Se divide el dataset en tres partes, manteniendo la misma proporción de comentarios tóxicos en cada una:
- Train: 700 filas (70%) — para entrenar
- Val: 150 filas (15%) — para ajustar hiperparámetros y monitorizar durante el entrenamiento
- Test: 150 filas (15%) — para la evaluación final (no se toca hasta el final)

La estratificación por `IsToxic` garantiza que si el 40% del dataset es tóxico, el 40% de cada split también lo sea.

**Ejecutar:**
```bash
uv run python src/data/prepare_dataset.py
```

---

## 4. Modelo 1 — TF-IDF + Logistic Regression (`src/model/train.py`)

### ¿Qué es TF-IDF?

TF-IDF (Term Frequency - Inverse Document Frequency) convierte texto en números. Para cada comentario, genera un vector donde cada posición representa una palabra (o par de palabras) y su valor indica qué tan importante es esa palabra en ese comentario comparado con todos los demás.

- **TF** (Term Frequency): cuántas veces aparece la palabra en este comentario
- **IDF** (Inverse Document Frequency): penaliza palabras que aparecen en muchos comentarios (son poco informativas)

Resultado: cada comentario se convierte en un vector de 10.000 números.

Configuración usada:
- `max_features=10000` — solo las 10.000 palabras/pares más frecuentes
- `ngram_range=(1, 2)` — unigramas (`hate`) y bigramas (`hate you`) — los bigramas capturan contexto

### ¿Qué es Logistic Regression?

Es un clasificador binario (0 o 1). Aprende qué combinaciones de palabras del vector TF-IDF predicen mejor cada etiqueta.

Se usa `class_weight="balanced"` porque el dataset está desbalanceado — sin esto, el modelo aprendería a predecir siempre 0 y tendría alta accuracy pero no detectaría nada.

### ¿Por qué un clasificador por etiqueta?

Multi-label no significa entrenar un solo modelo con 12 salidas. Aquí se entrena **un clasificador independiente por cada etiqueta**. Cada uno aprende qué palabras predicen su etiqueta específica.

### El problema de las etiquetas raras

Algunas etiquetas (`IsNationalist`, `IsSexist`, `IsHomophobic`, `IsRadicalism`) tienen tan pocos ejemplos positivos en 700 filas de train que `LogisticRegression` no puede aprender nada útil. Para estas, se usa un `DummyClassifier(strategy="most_frequent")` que simplemente predice siempre 0 — es honesto: no finge aprender algo que no puede.

### La clase `MultiLabelPipeline`

Agrupa el vectorizador TF-IDF y los 12 clasificadores en un solo objeto que se puede guardar y cargar con `joblib`. Tiene un método `predict(X)` que:
1. Transforma el texto con TF-IDF
2. Pasa el vector por cada uno de los 12 clasificadores
3. Devuelve una matriz de shape `(n_comentarios, 12)` con 0s y 1s

**¿Por qué está definida a nivel de módulo y no dentro de `main()`?**
`joblib` necesita poder encontrar la clase cuando carga el modelo guardado. Si estuviera dentro de `main()`, al cargar el modelo en otro script no la encontraría y daría error.

**Ejecutar:**
```bash
uv run python src/model/train.py
```

Guarda el modelo en `models/tfidf_lr.joblib`.

---

## 5. Evaluación TF-IDF + LR — `src/model/evaluate.py`

Carga el modelo guardado y el test set, predice y muestra un `classification_report` por cada etiqueta.

El `classification_report` muestra para cada etiqueta:

| Métrica | Qué significa |
|---|---|
| **Precision** | De los que predije como positivos, ¿cuántos lo eran realmente? |
| **Recall** | De todos los positivos reales, ¿cuántos detecté? |
| **F1** | Media armónica de precision y recall — el balance entre ambas |
| **Support** | Cuántos ejemplos reales hay de esa clase en el test set |

**Ejecutar:**
```bash
uv run python src/model/evaluate.py
```

---

## 6. Modelo 2 — DistilBERT fine-tuned (`models/distilbert_youtoxic/`)

### ¿Qué es DistilBERT?

DistilBERT es una versión comprimida de BERT (Bidirectional Encoder Representations from Transformers). Es un modelo de lenguaje pre-entrenado en millones de textos que ya "entiende" el inglés — conoce el significado de las palabras, el contexto, la ironía, etc.

A diferencia de TF-IDF, DistilBERT no trata el texto como una bolsa de palabras. Entiende que "I don't hate you" es diferente de "I hate you" aunque compartan palabras.

### ¿Qué es fine-tuning?

El modelo pre-entrenado sabe inglés pero no sabe clasificar toxicidad. El fine-tuning consiste en continuar el entrenamiento con nuestro dataset específico para que aprenda la tarea concreta. Se añade una capa de clasificación encima del modelo base y se entrena todo junto.

### Estado actual

Entrenado en **binario** (`IsToxic` solo, `num_labels=2`) usando el notebook `train_distilbert_colab.ipynb` en Google Colab. El plan es actualizarlo a multi-label con las 12 etiquetas para poder hacer el ensemble.

**Descargar:**
```bash
hf download KariRomero/distilbert-youtoxic --local-dir models/distilbert_youtoxic
```

---

## 7. Evaluación DistilBERT — `src/model/evaluate_distilbert.py`

Carga el modelo y el tokenizador desde `models/distilbert_youtoxic/`, tokeniza los textos del test set, pasa por el modelo y obtiene predicciones.

Muestra `classification_report` y matriz de confusión para `IsToxic`.

La **matriz de confusión** desglosa los errores:
```
                  Predicho: No tóxico   Predicho: Tóxico
Real: No tóxico   Verdaderos negativos  Falsos positivos
Real: Tóxico      Falsos negativos      Verdaderos positivos
```

- **Falsos positivos**: comentarios normales marcados como tóxicos (molesto para el usuario)
- **Falsos negativos**: comentarios tóxicos que se escapan (peligroso para la plataforma)

**Ejecutar:**
```bash
uv run python src/model/evaluate_distilbert.py
```

---

## 8. Resultados comparativos (IsToxic)

| Modelo | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| TF-IDF + LR | 72% | 70% | 70% | 70% |
| DistilBERT | **77%** | **73%** | **78%** | 68% |

- DistilBERT tiene mejor accuracy y precision — cuando dice que algo es tóxico, acierta más
- TF-IDF + LR tiene mejor recall — detecta más comentarios tóxicos (menos falsos negativos)
- El ensemble busca combinar lo mejor de ambos

---

## 9. Plan: Ensemble

El objetivo final es combinar ambos modelos. La idea es:

1. Ambos modelos predicen las 12 etiquetas (probabilidades, no solo 0/1)
2. Se hace un promedio ponderado dando más peso a DistilBERT (mejor modelo)
3. Si la probabilidad combinada supera un umbral (ej. 0.5), se predice 1

Esto suele mejorar los resultados porque los dos modelos cometen errores distintos — TF-IDF falla en contexto, DistilBERT puede fallar en palabras muy específicas del dominio.

---

## 10. Modelos en Hugging Face

Los modelos entrenados están publicados para no tener que reentrenar desde cero:

- TF-IDF + LR: [KariRomero/tfidf-lr-youtoxic](https://huggingface.co/KariRomero/tfidf-lr-youtoxic)
- DistilBERT: [KariRomero/distilbert-youtoxic](https://huggingface.co/KariRomero/distilbert-youtoxic)

```bash
# Descargar TF-IDF + LR
hf download KariRomero/tfidf-lr-youtoxic tfidf_lr.joblib --local-dir models/

# Descargar DistilBERT
hf download KariRomero/distilbert-youtoxic --local-dir models/distilbert_youtoxic
```

---

## 11. Flujo completo de principio a fin

```
youtoxic_english_1000.csv
        ↓
prepare_dataset.py          → limpia texto, convierte etiquetas, hace split
        ↓
train.csv / val.csv / test.csv
        ↓
train.py                    → TF-IDF + 12 LR → tfidf_lr.joblib
train_distilbert_colab.ipynb → fine-tuning DistilBERT → distilbert_youtoxic/
        ↓
evaluate.py                 → métricas TF-IDF + LR en test set
evaluate_distilbert.py      → métricas DistilBERT en test set
        ↓
(pendiente) ensemble        → combina ambos modelos
        ↓
streamlit_app.py            → interfaz web para probar el clasificador
```
