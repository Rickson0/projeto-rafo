import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="rafo_explorer")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return alt, mo, pl


# ══════════════════════════════════════════════════════════════════════
# 1. Carregar assets pré-computados
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(mo, pl):
    # No WASM, `notebook_location` aponta para o diretório do index.html.
    _base = mo.notebook_location() / "public"

    topic_info      = pl.read_parquet(str(_base / "topic_info.parquet"))
    topic_palavras  = pl.read_parquet(str(_base / "topic_palavras.parquet"))
    topics_ot       = pl.read_parquet(str(_base / "topics_over_time.parquet"))
    doc_topicos     = pl.read_parquet(str(_base / "doc_topicos.parquet"))
    nmf_topicos     = pl.read_parquet(str(_base / "nmf_topicos.parquet"))
    dataset         = pl.read_parquet(str(_base / "dataset.parquet"))

    return (
        dataset, doc_topicos, nmf_topicos,
        topic_info, topic_palavras, topics_ot,
    )


# ══════════════════════════════════════════════════════════════════════
# 2. Header
# ══════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(dataset, mo):
    mo.md(f"""
    # 📚 RAFO Explorer

    Análise interativa das perguntas que o Brandon Sanderson responde com
    **RAFO** (*Read And Find Out*) em Q&As.

    **{len(dataset):,}** pares pergunta↔resposta pré-computados.
    """)
    return


# ══════════════════════════════════════════════════════════════════════
# 3. Controles
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(dataset, mo, pl):
    _anos = dataset["ano"].drop_nulls()

    ctrl_anos = mo.ui.range_slider(
        start=int(_anos.min()), stop=int(_anos.max()),
        value=(int(_anos.min()), int(_anos.max())), step=1,
        label="📅 Intervalo de anos",
    )
    ctrl_n = mo.ui.slider(
        start=3, stop=20, value=8, step=1,
        label="Nº de tópicos exibidos",
    )
    ctrl_busca = mo.ui.text(
        value="", label="🔎 Buscar termo na pergunta/resposta",
        full_width=True,
    )
    ctrl_temporal = mo.ui.checkbox(value=True, label="Mostrar evolução temporal")
    ctrl_tabela   = mo.ui.checkbox(value=True, label="Mostrar tabela de exemplos")

    mo.vstack([
        mo.hstack([ctrl_anos], justify="start"),
        mo.hstack([ctrl_n, ctrl_temporal, ctrl_tabela], justify="start", gap="2rem"),
        ctrl_busca,
    ])
    return ctrl_anos, ctrl_busca, ctrl_n, ctrl_tabela, ctrl_temporal


# ══════════════════════════════════════════════════════════════════════
# 4. Frequência por ano
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(alt, ctrl_anos, dataset, pl):
    _lo, _hi = ctrl_anos.value
    _freq = (
        dataset
        .filter(pl.col("ano").is_between(_lo, _hi))
        .group_by("ano").len().sort("ano")
    )

    alt.Chart(_freq).mark_bar().encode(
        x=alt.X("ano:O", title="Ano"),
        y=alt.Y("len:Q", title="Nº de RAFOs"),
        tooltip=["ano", "len"],
    ).properties(
        title="Frequência de RAFOs por ano",
        width="container", height=280,
    ).interactive()
    return


# ══════════════════════════════════════════════════════════════════════
# 5. Top palavras por tópico
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(alt, ctrl_n, topic_palavras):
    _top_topicos = (
        topic_palavras
        .group_by("topico").agg(pl.col("peso").sum().alias("peso_total"))
        .sort("peso_total", descending=True)
        .head(ctrl_n.value)["topico"].to_list()
    )

    _filtrado = topic_palavras.filter(pl.col("topico").is_in(_top_topicos))

    alt.Chart(_filtrado).mark_bar().encode(
        x=alt.X("peso:Q", title="Peso"),
        y=alt.Y("palavra:N", sort="-x", title=None),
        row=alt.Row("topico:N", title="Tópico", header=alt.Header(labelAngle=0)),
        tooltip=["topico", "palavra", "peso"],
    ).properties(
        title="Top palavras por tópico (BERTopic)",
        width="container", height=120,
    )
    return


# ══════════════════════════════════════════════════════════════════════
# 6. Evolução temporal
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(alt, ctrl_n, ctrl_temporal, topics_ot):
    _out = None
    if ctrl_temporal.value:
        _top = (
            topics_ot
            .group_by("Topic").agg(pl.col("Frequency").sum().alias("total"))
            .sort("total", descending=True)
            .head(ctrl_n.value)["Topic"].to_list()
        )
        _filtrado = topics_ot.filter(pl.col("Topic").is_in(_top))

        _out = alt.Chart(_filtrado).mark_line(point=True).encode(
            x=alt.X("Timestamp:O", title="Ano"),
            y=alt.Y("Frequency:Q", title="Frequência"),
            color=alt.Color("Topic:N", title="Tópico"),
            tooltip=["Topic", "Timestamp", "Frequency"],
        ).properties(
            title="Evolução temporal dos tópicos",
            width="container", height=380,
        ).interactive()
    _out
    return


# ══════════════════════════════════════════════════════════════════════
# 7. Tabela paginada com busca
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(ctrl_anos, ctrl_busca, ctrl_tabela, dataset, mo, pl):
    if not ctrl_tabela.value:
        mo.md("*Ative 'Mostrar tabela' acima para ver exemplos.*")
    else:
        _lo, _hi = ctrl_anos.value
        _filtrado = dataset.filter(pl.col("ano").is_between(_lo, _hi))

        if ctrl_busca.value.strip():
            _q = ctrl_busca.value.lower().strip()
            _filtrado = _filtrado.filter(
                pl.col("text_full").str.contains(_q, literal=True)
            )

        mo.ui.table(
            _filtrado
              .select("date", "ano", "speaker_p", "pergunta", "resposta")
              .sort("date", descending=True)
              .head(500),
            pagination=True,
            page_size=10,
            label=f"📋 Exemplos ({len(_filtrado):,} registros)",
        )
    return


# ══════════════════════════════════════════════════════════════════════
# 8. NMF (tópicos TF-IDF, estático)
# ══════════════════════════════════════════════════════════════════════

@app.cell
def _(mo, nmf_topicos):
    mo.accordion({
        "🧩 Tópicos NMF (TF-IDF)": mo.ui.table(
            nmf_topicos, pagination=False, label=None,
        ),
    })
    return


if __name__ == "__main__":
    app.run()