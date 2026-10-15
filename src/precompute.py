"""Pré-computa tudo que é pesado demais para rodar no WASM.

Rode sempre que os dados brutos mudarem:
    python scripts/precompute.py

Gera:
    data/derived/dataset.parquet        pares pergunta↔resposta
    data/derived/topic_info.parquet     tabela de tópicos do BERTopic
    data/derived/topics_over_time.parquet
    data/derived/freq_por_ano.parquet
"""
from pathlib import Path

import polars as pl
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "processed" / "wobs_limpos.parquet"
OUT = ROOT / "data" / "derived"
OUT.mkdir(parents=True, exist_ok=True)

# ── importa a lógica que já existe no app ────────────────────────────────
# (por enquanto copie/cole as funções aqui; depois extraia para rafo/loader.py)
from rafo.pipeline import construir_dataset_rafo, limpar_texto  # noqa: E402

def main() -> None:
    df = pl.read_parquet(RAW)
    dataset = construir_dataset_rafo(df)
    dataset = limpar_texto(dataset)
    dataset = dataset.with_columns(
        text_full=pl.col("pergunta_limpa") + " " + pl.col("resposta_limpa"),
        ano=pl.col("date").dt.year(),
    )
    dataset.write_parquet(OUT / "dataset.parquet")

    # frequência por ano
    (
        dataset.group_by("ano").len().sort("ano")
        .write_parquet(OUT / "freq_por_ano.parquet")
    )

    # BERTopic — treina com vocabulário Cosmere
    vocab = _carregar_vocabulario()  # importe do app ou duplique aqui
    vec = CountVectorizer(vocabulary=vocab)
    model = BERTopic(
        language="english",
        calculate_probabilities=True,
        vectorizer_model=vec,
        nr_topics=8,
        verbose=True,
    )
    topics, _ = model.fit_transform(dataset["text_full"].to_list())

    (
        model.get_topic_info()
        .to_pandas()
        .pipe(pl.from_pandas)
        .write_parquet(OUT / "topic_info.parquet")
    )

    tot = model.topics_over_time(
        dataset["text_full"].to_list(),
        timestamps=dataset["ano"].to_list(),
        nr_bins=10,
    )
    tot.to_pandas().pipe(pl.from_pandas).write_parquet(
        OUT / "topics_over_time.parquet"
    )

    # guarda os tópicos atribuídos a cada linha (para filtrar no client)
    (
        dataset.with_columns(topico=pl.Series(topics))
        .write_parquet(OUT / "dataset_com_topicos.parquet")
    )

    print(f"✓ artefatos em {OUT}")


def _carregar_vocabulario() -> set[str]:
    """Isolado aqui para não arrastar marimo para dentro do CI."""
    # copie o dict do app.py ou importe de rafo/vocabulario.py
    from rafo.vocabulario import vocabulario
    return vocabulario


if __name__ == "__main__":
    main()