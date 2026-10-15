import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="RAFO Explorer")


# ══════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return alt, mo, pl


# ══════════════════════════════════════════════════════════════════════
# Carregamento dos assets (JSON no WASM, parquet local)
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(mo, pl):
    import sys
    import json

    _em_wasm = "pyodide" in sys.modules or "emscripten" in sys.platform

    if _em_wasm:
        from urllib.request import urlopen

        _base = str(mo.notebook_location()).rstrip("/")
        if _base.endswith(".html"):
            _base = _base.rsplit("/", 1)[0]

        def _ler(nome: str) -> pl.DataFrame:
            url = f"{_base}/public/{nome}.json"
            with urlopen(url) as resp:
                texto = resp.read().decode("utf-8")
            return pl.DataFrame(json.loads(texto))
    else:
        _base = str(mo.notebook_location() / "public")

        def _ler(nome: str) -> pl.DataFrame:
            return pl.read_parquet(f"{_base}/{nome}.parquet")

    topic_info     = _ler("topic_info")
    topic_palavras = _ler("topic_palavras")
    topics_ot      = _ler("topics_over_time")
    doc_topicos    = _ler("doc_topicos")
    nmf_topicos    = _ler("nmf_topicos")
    dataset        = _ler("dataset")

    return dataset, doc_topicos, nmf_topicos, topic_info, topic_palavras, topics_ot


# ══════════════════════════════════════════════════════════════════════
# Limpeza dos nomes de tópicos (BERTopic gera nomes feios tipo "0_x_y_z")
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(pl, topic_info):
    def _nome_legivel(topico: int, rep) -> str:
        if topico == -1:
            return "❓ Outliers"
        palavras = [p for p in (rep or []) if p and not p.startswith("-")]
        if not palavras:
            return f"Tópico {topico}"
        return " · ".join(p.capitalize() for p in palavras[:4])

    topic_info_limpo = (
        topic_info
        .with_columns(
            pl.struct(["Topic", "Representation"])
              .map_elements(
                  lambda s: _nome_legivel(s["Topic"], s["Representation"]),
                  return_dtype=pl.Utf8,
              )
              .alias("label"),
            pl.col("Count").alias("n_docs"),
        )
        .sort("n_docs", descending=True)
    )
    return (topic_info_limpo,)


# ══════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(dataset, mo):
    mo.md(f"""
    # 📚 RAFO Explorer

    Análise interativa das perguntas que o Brandon Sanderson responde com
    **RAFO** (_Read And Find Out_) em Q&As e sessões de autógrafos.

    **{len(dataset):,}** pares pergunta↔resposta pré-computados.
    """)
    return


# ══════════════════════════════════════════════════════════════════════
# Controles
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(dataset, mo):
    _anos = dataset["ano"].drop_nulls()

    ctrl_anos = mo.ui.range_slider(
        start=int(_anos.min()), stop=int(_anos.max()),
        value=(int(_anos.min()), int(_anos.max())), step=1,
        label="📅 Intervalo de anos",
    )
    ctrl_n = mo.ui.slider(
        start=3, stop=15, value=6, step=1,
        label="📊 Nº de tópicos exibidos",
    )
    ctrl_busca = mo.ui.text(
        value="", label="🔎 Buscar termo na pergunta/resposta",
        full_width=True,
    )
    ctrl_temporal = mo.ui.checkbox(value=True, label="Mostrar evolução temporal")
    ctrl_tabela   = mo.ui.checkbox(value=True, label="Mostrar tabela de exemplos")

    mo.vstack([
        mo.hstack([ctrl_anos], justify="start"),
        mo.hstack([ctrl_n, ctrl_temporal, ctrl_tabela],
                  justify="start", gap="2rem"),
        ctrl_busca,
    ])
    return ctrl_anos, ctrl_busca, ctrl_n, ctrl_tabela, ctrl_temporal


# ══════════════════════════════════════════════════════════════════════
# Frequência de RAFOs por ano
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(alt, ctrl_anos, dataset, pl):
    _lo, _hi = ctrl_anos.value
    _freq = (
        dataset
        .filter(pl.col("ano").is_between(_lo, _hi))
        .group_by("ano").len().sort("ano")
    )

    alt.Chart(_freq).mark_bar(color="#4c78a8").encode(
        x=alt.X("ano:O", title="Ano", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("len:Q", title="Nº de RAFOs"),
        tooltip=[
            alt.Tooltip("ano:O", title="Ano"),
            alt.Tooltip("len:Q", title="RAFOs"),
        ],
    ).properties(
        title="📈 Frequência de RAFOs por ano",
        width="container", height=280,
    ).interactive()
    return


# ══════════════════════════════════════════════════════════════════════
# Top palavras por tópico (barchart legível)
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(alt, ctrl_n, pl, topic_info_limpo, topic_palavras):
    _top = (
        topic_info_limpo
        .filter(pl.col("Topic") != -1)
        .head(ctrl_n.value)
        .select("Topic", "label")
    )

    _filtrado = (
        topic_palavras
        .filter(pl.col("topico").is_in(_top["Topic"].to_list()))
        .join(_top, left_on="topico", right_on="Topic", how="inner")
    )

    alt.Chart(_filtrado).mark_bar(color="#e45756").encode(
        x=alt.X("peso:Q", title="Peso", axis=alt.Axis(format=".3f")),
        y=alt.Y(
            "palavra:N", sort="-x", title=None,
            axis=alt.Axis(labelFontSize=11),
        ),
        row=alt.Row(
            "label:N",
            title=None,
            header=alt.Header(
                labelAngle=0, labelAlign="left",
                labelFontSize=12, labelFontWeight="bold",
            ),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Tópico"),
            alt.Tooltip("palavra:N", title="Palavra"),
            alt.Tooltip("peso:Q", title="Peso", format=".3f"),
        ],
    ).properties(
        title="🧩 Top palavras por tópico (BERTopic)",
        width="container", height=110,
    ).resolve_scale(y="independent")
    return


# ══════════════════════════════════════════════════════════════════════
# Evolução temporal dos tópicos
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(alt, ctrl_n, ctrl_temporal, pl, topic_info_limpo, topics_ot):
    if not ctrl_temporal.value:
        _out = None
    else:
        _top_ids = (
            topic_info_limpo
            .filter(pl.col("Topic") != -1)
            .head(ctrl_n.value)["Topic"].to_list()
        )
        _filtrado = (
            topics_ot
            .filter(pl.col("Topic").is_in(_top_ids))
            .join(
                topic_info_limpo.select("Topic", "label"),
                on="Topic", how="left",
            )
        )

        _out = alt.Chart(_filtrado).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("Timestamp:O", title="Ano",
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Frequency:Q", title="Frequência"),
            color=alt.Color("label:N", title="Tópico",
                            legend=alt.Legend(orient="bottom", columns=2)),
            tooltip=[
                alt.Tooltip("label:N", title="Tópico"),
                alt.Tooltip("Timestamp:O", title="Ano"),
                alt.Tooltip("Frequency:Q", title="Freq."),
            ],
        ).properties(
            title="🕒 Evolução temporal dos tópicos",
            width="container", height=380,
        ).interactive()
    _out
    return


# ══════════════════════════════════════════════════════════════════════
# Tabela de tópicos identificados
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(mo, pl, topic_info_limpo):
    mo.ui.table(
        topic_info_limpo
        .filter(pl.col("Topic") != -1)
        .select(
            pl.col("Topic").alias("#"),
            pl.col("label").alias("Tópico"),
            pl.col("n_docs").alias("Docs"),
        ),
        pagination=False,
        selection=None,
        label="📋 Tópicos identificados",
    )
    return


# ══════════════════════════════════════════════════════════════════════
# Tabela paginada com busca
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(ctrl_anos, ctrl_busca, ctrl_tabela, dataset, mo, pl):
    if not ctrl_tabela.value:
        _out = mo.md("*Ative 'Mostrar tabela' acima para ver exemplos.*")
    else:
        _lo, _hi = ctrl_anos.value
        _filtrado = dataset.filter(pl.col("ano").is_between(_lo, _hi))

        if ctrl_busca.value.strip():
            _q = ctrl_busca.value.lower().strip()
            _filtrado = _filtrado.filter(
                pl.col("text_full").str.contains(_q, literal=True)
            )

        _out = mo.ui.table(
            _filtrado
            .select("date", "ano", "speaker_p", "pergunta", "resposta")
            .sort("date", descending=True)
            .head(500),
            pagination=True,
            page_size=10,
            label=f"💬 Exemplos de RAFO ({len(_filtrado):,} registros)",
        )
    _out
    return


# ══════════════════════════════════════════════════════════════════════
# Tópicos NMF (TF-IDF) — accordion colapsável
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(mo, nmf_topicos):
    mo.accordion({
        "🧬 Tópicos NMF (TF-IDF) — visão alternativa": mo.ui.table(
            nmf_topicos,
            pagination=False,
            selection=None,
        ),
    })
    return


if __name__ == "__main__":
    app.run()