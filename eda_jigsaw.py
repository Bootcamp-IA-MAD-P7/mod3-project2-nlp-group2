"""
EDA del dataset Jigsaw Toxic Comment Classification Challenge.

Uso.
    python eda_jigsaw.py
    python eda_jigsaw.py --train ruta/al/train.csv

Que hace.
    Carga train.csv, calcula todas las metricas del EDA (estructura, calidad,
    distribucion de etiquetas, cardinalidad multietiqueta, coocurrencia,
    longitud, rasgos de texto y vocabulario por clase) y escribe un informe
    HTML con las figuras incrustadas en reports/eda_jigsaw_report.html.
    Las figuras sueltas quedan tambien en reports/figures/.

Dependencias.
    pip install pandas numpy matplotlib
    (no usa NLTK ni descargas, las stopwords van embebidas)
"""

import argparse
import base64
import io
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin ventana, para generar imagenes en disco
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COL = "comment_text"

# Paleta oficial de YouTube. Rojo para la clase positiva, oscuro para el resto.
YT_RED = "#FF0000"
YT_DARK = "#282828"
YT_GRAY = "#909090"
YT_WHITE = "#FFFFFF"

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with about against between into through during
before after above below from up down out off over under again further once here there all any both each
few more most other some such no nor not only own same so than too very s t can will just don don't should
now i me my myself we our ours you your yours he him his she her it its they them their what which who whom
this that these those am is are was were be been being have has had do does did doing would could ought i'm
you're he's she's it's we're they're i've you've we've they've isn't aren't wasn't weren't hasn't haven't
hadn't doesn't didn't won't wouldn't shan't shouldn't couldn't mustn't as also get got like one via
""".split())

URL_RE = re.compile(r"http\S+|www\.\S+")
TOKEN_RE = re.compile(r"[a-z]+")


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def find_train(explicit):
    """Localiza train.csv en rutas habituales o usa la ruta indicada."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"No existe el archivo indicado, {p}")
    candidates = [
        "data/raw/jigsaw/train.csv",
        "data/raw/train.csv",
        "data/train.csv",
        "train.csv",
    ]
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    globbed = list(Path("data").rglob("train.csv")) if Path("data").exists() else []
    if globbed:
        return globbed[0]
    raise FileNotFoundError(
        "No encuentro train.csv. Descargalo de la competicion Jigsaw y ponlo en "
        "data/raw/jigsaw/train.csv, o pasa la ruta con --train."
    )


def style_ax(ax):
    """Aplica el estilo comun a un eje."""
    ax.set_facecolor(YT_WHITE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=YT_DARK)
    ax.title.set_color(YT_DARK)
    ax.xaxis.label.set_color(YT_DARK)
    ax.yaxis.label.set_color(YT_DARK)


def fig_to_html(fig, filename):
    """Guarda la figura en disco y devuelve una etiqueta img con la imagen embebida."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / filename, dpi=150, bbox_inches="tight", facecolor=YT_WHITE)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=YT_WHITE)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{filename}"/>'


def df_to_html(df, index=False):
    return df.to_html(classes="tbl", border=0, index=index, escape=True)


def tokens(text):
    text = URL_RE.sub(" ", str(text).lower())
    return [w for w in TOKEN_RE.findall(text) if w not in STOPWORDS and len(w) > 2]


# ----------------------------------------------------------------------------
# Carga y rasgos de texto
# ----------------------------------------------------------------------------
def load_and_feature(train_path):
    df = pd.read_csv(train_path)
    missing = [c for c in LABELS + [TEXT_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas esperadas, {missing}. Columnas encontradas, {list(df.columns)}"
        )
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    df["n_labels"] = df[LABELS].sum(axis=1)
    df["any_toxic"] = (df["n_labels"] > 0).astype(int)
    df["char_len"] = df[TEXT_COL].str.len()
    df["word_len"] = df[TEXT_COL].str.split().apply(len)
    letters = df[TEXT_COL].str.count(r"[A-Za-z]").replace(0, np.nan)
    df["upper_ratio"] = (df[TEXT_COL].str.count(r"[A-Z]") / letters).fillna(0)
    df["excl"] = df[TEXT_COL].str.count("!")
    df["quest"] = df[TEXT_COL].str.count(r"\?")
    df["has_url"] = df[TEXT_COL].str.contains(URL_RE).astype(int)
    return df


# ----------------------------------------------------------------------------
# Bloques del informe
# ----------------------------------------------------------------------------
def block_intro(df, train_path):
    n, c = df.shape
    txt = (
        "Este informe explora el dataset Jigsaw Toxic Comment Classification Challenge, "
        "comentarios de paginas de discusion de Wikipedia en ingles etiquetados por humanos. "
        "El objetivo es entender su estructura y sus etiquetas antes de decidir preprocesado, "
        "definicion del target y balanceo. Cada bloque enuncia una pregunta, muestra la evidencia "
        "y la lee en una frase."
    )
    meta = pd.DataFrame(
        {"propiedad": ["archivo", "filas", "columnas"], "valor": [str(train_path), n, c]}
    )
    return section("Contexto", txt, table=df_to_html(meta))


def block_structure(df):
    txt = (
        "Miramos forma, tipos y primeras filas para confirmar el esquema real y no trabajar "
        "sobre supuestos. Son seis etiquetas binarias independientes y una columna de texto."
    )
    head = df[[TEXT_COL] + LABELS].head(5).copy()
    head[TEXT_COL] = head[TEXT_COL].str.slice(0, 90) + "..."
    dtypes = pd.DataFrame({"columna": df.columns, "tipo": df.dtypes.astype(str).values})
    html = "<h3>Primeras filas</h3>" + df_to_html(head)
    html += "<h3>Tipos de dato</h3>" + df_to_html(dtypes)
    return section("Estructura y head", txt, extra_html=html)


def block_quality(df):
    txt = (
        "Revisamos nulos, duplicados y comentarios vacios. Un comentario repetido infla el peso "
        "de ciertos ejemplos y puede filtrarse entre entrenamiento y validacion, asi que conviene "
        "detectarlo antes de partir los datos."
    )
    nulls = df.isna().sum()
    quality = pd.DataFrame(
        {
            "chequeo": [
                "nulos en texto",
                "nulos totales",
                "duplicados exactos de comentario",
                "comentarios vacios",
            ],
            "valor": [
                int(nulls.get(TEXT_COL, 0)),
                int(nulls.sum()),
                int(df[TEXT_COL].duplicated().sum()),
                int((df["char_len"] == 0).sum()),
            ],
        }
    )
    return section("Calidad de los datos", txt, table=df_to_html(quality))


def block_label_dist(df):
    txt = (
        "El dataset es multietiqueta, cada comentario puede llevar varias marcas a la vez. Vemos "
        "que proporcion de positivos aporta cada etiqueta. Un desbalanceo fuerte obliga a evaluar "
        "con F1 y con precision recall en lugar de accuracy, que premiaria predecir siempre la "
        "clase mayoritaria."
    )
    counts = df[LABELS].sum().sort_values(ascending=False)
    pct = (counts / len(df) * 100).round(2)
    tbl = pd.DataFrame({"etiqueta": counts.index, "positivos": counts.values, "porcentaje": pct.values})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color=YT_RED)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{pct.values[i]}%", ha="center", va="bottom", color=YT_DARK, fontsize=9)
    ax.set_title("Positivos por etiqueta")
    ax.set_ylabel("numero de comentarios")
    plt.xticks(rotation=20)
    style_ax(ax)
    img = fig_to_html(fig, "01_positivos_por_etiqueta.png")

    any_pct = round(df["any_toxic"].mean() * 100, 2)
    read = (
        f"El target binario que marca positivo si hay al menos una etiqueta cubre el {any_pct}% "
        "de los comentarios, el resto son limpios. El desbalanceo es marcado en todas las etiquetas."
    )
    return section("Distribucion de etiquetas", txt, table=df_to_html(tbl), img=img, read=read)


def block_cardinality(df):
    txt = (
        "Contamos cuantas etiquetas lleva cada comentario. Esto mide la cardinalidad multietiqueta "
        "e informa si conviene modelar las etiquetas de forma conjunta o por separado."
    )
    card = df["n_labels"].value_counts().sort_index()
    tbl = pd.DataFrame({"n_etiquetas": card.index, "comentarios": card.values})

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [YT_DARK if k == 0 else YT_RED for k in card.index]
    ax.bar(card.index.astype(str), card.values, color=colors)
    ax.set_title("Numero de etiquetas por comentario")
    ax.set_xlabel("etiquetas activas")
    ax.set_ylabel("comentarios")
    style_ax(ax)
    img = fig_to_html(fig, "02_cardinalidad.png")

    combos = (
        df[df["any_toxic"] == 1][LABELS]
        .apply(lambda r: ", ".join([l for l in LABELS if r[l] == 1]), axis=1)
        .value_counts()
        .head(10)
        .reset_index()
    )
    combos.columns = ["combinacion de etiquetas", "comentarios"]
    extra = "<h3>Combinaciones de etiquetas mas frecuentes</h3>" + df_to_html(combos)
    return section("Cardinalidad multietiqueta", txt, table=df_to_html(tbl), img=img, extra_html=extra)


def block_cooccurrence(df):
    txt = (
        "Analizamos que etiquetas aparecen juntas. Estudiamos la coocurrencia en conteos y la "
        "correlacion phi entre etiquetas binarias. Correlaciones altas indican solapamiento "
        "conceptual, por ejemplo entre insult y toxic, y sugieren que un modelo multietiqueta "
        "puede compartir senal."
    )
    X = df[LABELS].values
    co = X.T @ X
    co_df = pd.DataFrame(co, index=LABELS, columns=LABELS)

    fig1, ax1 = plt.subplots(figsize=(6.5, 5.5))
    im1 = ax1.imshow(co_df.values, cmap="Reds")
    ax1.set_xticks(range(len(LABELS)))
    ax1.set_yticks(range(len(LABELS)))
    ax1.set_xticklabels(LABELS, rotation=45, ha="right")
    ax1.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax1.text(j, i, int(co_df.values[i, j]), ha="center", va="center",
                     color=YT_DARK, fontsize=8)
    ax1.set_title("Coocurrencia de etiquetas (conteos)")
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    img1 = fig_to_html(fig1, "03_coocurrencia.png")

    corr = df[LABELS].corr()
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    im2 = ax2.imshow(corr.values, cmap="Reds", vmin=0, vmax=1)
    ax2.set_xticks(range(len(LABELS)))
    ax2.set_yticks(range(len(LABELS)))
    ax2.set_xticklabels(LABELS, rotation=45, ha="right")
    ax2.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax2.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color=YT_DARK, fontsize=8)
    ax2.set_title("Correlacion phi entre etiquetas")
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    img2 = fig_to_html(fig2, "04_correlacion.png")

    return section("Analisis bivariado de etiquetas", txt, img=img1 + img2)


def block_length(df):
    txt = (
        "Comparamos la longitud de los comentarios toxicos y limpios. Ademas de describir el "
        "corpus, esto fija la longitud maxima de secuencia al tokenizar para un transformer, donde "
        "truncar mal pierde informacion."
    )
    grp = df.groupby("any_toxic")["word_len"]
    summ = grp.agg(mediana="median", media="mean",
                   p95=lambda s: s.quantile(0.95)).round(1).reset_index()
    summ["any_toxic"] = summ["any_toxic"].map({0: "limpio", 1: "toxico"})

    clean = df[df["any_toxic"] == 0]["word_len"].clip(upper=df["word_len"].quantile(0.99))
    toxic = df[df["any_toxic"] == 1]["word_len"].clip(upper=df["word_len"].quantile(0.99))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bp = ax.boxplot([clean, toxic], patch_artist=True, showfliers=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["limpio", "toxico"])
    for patch, col in zip(bp["boxes"], [YT_DARK, YT_RED]):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color(YT_WHITE)
    ax.set_title("Longitud en palabras por clase")
    ax.set_ylabel("palabras por comentario")
    style_ax(ax)
    img = fig_to_html(fig, "05_longitud_por_clase.png")
    return section("Longitud de los comentarios", txt, table=df_to_html(summ), img=img)


def block_textfeatures(df):
    txt = (
        "Cruzamos rasgos superficiales del texto con el target. Si la proporcion de mayusculas o "
        "los signos de exclamacion separan clases, son candidatos a features o a pasos de "
        "normalizacion en el preprocesado."
    )
    feats = ["upper_ratio", "excl", "quest", "has_url", "char_len"]
    summ = df.groupby("any_toxic")[feats].mean().round(3).reset_index()
    summ["any_toxic"] = summ["any_toxic"].map({0: "limpio", 1: "toxico"})

    means = df.groupby("any_toxic")[["upper_ratio", "excl", "quest"]].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(3)
    w = 0.38
    ax.bar(x - w / 2, means.loc[0].values, width=w, label="limpio", color=YT_DARK)
    ax.bar(x + w / 2, means.loc[1].values, width=w, label="toxico", color=YT_RED)
    ax.set_xticks(x)
    ax.set_xticklabels(["prop. mayusculas", "exclamaciones", "interrogaciones"])
    ax.set_title("Media de rasgos de texto por clase")
    ax.legend(frameon=False)
    style_ax(ax)
    img = fig_to_html(fig, "06_rasgos_por_clase.png")
    return section("Rasgos de texto frente al target", txt, table=df_to_html(summ), img=img)


def block_vocab(df):
    txt = (
        "Listamos los terminos mas frecuentes por clase tras quitar stopwords en ingles. Da una "
        "primera intuicion del lenguaje asociado a cada clase y orienta la limpieza y la eleccion "
        "de vectorizacion."
    )
    top_toxic = Counter()
    for t in df[df["any_toxic"] == 1][TEXT_COL].head(20000):
        top_toxic.update(tokens(t))
    top_clean = Counter()
    for t in df[df["any_toxic"] == 0][TEXT_COL].head(20000):
        top_clean.update(tokens(t))

    def bar(counter, title, fname, color):
        common = counter.most_common(20)[::-1]
        words = [w for w, _ in common]
        vals = [v for _, v in common]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(words, vals, color=color)
        ax.set_title(title)
        style_ax(ax)
        return fig_to_html(fig, fname)

    img1 = bar(top_toxic, "Top terminos, clase toxica", "07_top_toxico.png", YT_RED)
    img2 = bar(top_clean, "Top terminos, clase limpia", "08_top_limpio.png", YT_DARK)
    note = (
        "<p class='read'>Nota. El conteo de vocabulario se limita a una muestra de comentarios por "
        "clase para que el informe corra rapido. Sube el limite en el codigo si quieres el corpus "
        "completo.</p>"
    )
    return section("Vocabulario por clase", txt, img=img1 + img2, extra_html=note)


# ----------------------------------------------------------------------------
# Ensamblado del HTML
# ----------------------------------------------------------------------------
def section(title, text, table=None, img=None, read=None, extra_html=None):
    parts = [f"<section><h2>{title}</h2>", f"<p>{text}</p>"]
    if table:
        parts.append(table)
    if img:
        parts.append(f"<div class='figs'>{img}</div>")
    if extra_html:
        parts.append(extra_html)
    if read:
        parts.append(f"<p class='read'>{read}</p>")
    parts.append("</section>")
    return "\n".join(parts)


CSS = """
body { font-family: Arial, Helvetica, sans-serif; color: #282828; background: #FFFFFF;
       max-width: 980px; margin: 0 auto; padding: 32px; line-height: 1.5; }
h1 { color: #282828; border-bottom: 4px solid #FF0000; padding-bottom: 8px; }
h2 { color: #282828; margin-top: 40px; border-left: 5px solid #FF0000; padding-left: 10px; }
h3 { color: #282828; margin-top: 20px; }
p { color: #282828; }
p.read { background: #f5f5f5; border-left: 3px solid #FF0000; padding: 8px 12px; }
.figs img { max-width: 100%; height: auto; margin: 12px 0; border: 1px solid #eee; }
table.tbl { border-collapse: collapse; margin: 12px 0; font-size: 14px; }
table.tbl th { background: #282828; color: #FFFFFF; padding: 6px 10px; text-align: left; }
table.tbl td { border-bottom: 1px solid #eee; padding: 6px 10px; }
"""


def build_report(df, train_path):
    blocks = [
        block_intro(df, train_path),
        block_structure(df),
        block_quality(df),
        block_label_dist(df),
        block_cardinality(df),
        block_cooccurrence(df),
        block_length(df),
        block_textfeatures(df),
        block_vocab(df),
    ]
    html = (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<title>EDA Jigsaw Toxic Comment</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>EDA. Jigsaw Toxic Comment Classification Challenge</h1>"
        + "\n".join(blocks)
        + "</body></html>"
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "eda_jigsaw_report.html"
    out.write_text(html, encoding="utf-8")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="EDA de Jigsaw Toxic Comment")
    ap.add_argument("--train", default=None, help="ruta a train.csv")
    args = ap.parse_args()

    train_path = find_train(args.train)
    print(f"[1/3] Cargando {train_path}")
    df = load_and_feature(train_path)
    print(f"      {len(df)} filas, {df.shape[1]} columnas")
    print(f"      positivos (al menos una etiqueta), {df['any_toxic'].mean()*100:.2f}%")

    print("[2/3] Calculando figuras e informe")
    out = build_report(df, train_path)

    print(f"[3/3] Listo. Abre {out}")
    print(f"      figuras sueltas en {FIGDIR}")


if __name__ == "__main__":
    main()
