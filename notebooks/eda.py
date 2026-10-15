import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import json
    import pickle

    return json, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Exploração dos Dados brutos
    """)
    return


@app.cell
def _(json):
    with open('data/raw/wobs.json', 'r',encoding='utf-8') as f:
       raw_text = json.load(f)
   
    return


@app.cell
def _(pl):
    df = pl.read_json('data/raw/wobs.json').sort('event_date')
    df.write_parquet('data/raw/wobs.parquet')
    return (df,)


@app.cell
def _(df, pl):
    df.schema
    df.null_count()
    df.select(pl.all().n_unique())
    df.sample(5).to_dicts()
    return


@app.cell
def _(df):
    df_tratado = (
        df.explode('lines')
            .unnest('lines')
    
    )
    df_tratado
    return (df_tratado,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Verificação de valores nulos
    """)
    return


@app.cell
def _(df_tratado):
    df_tratado.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    #Remoção de duplicatas
    """)
    return


@app.cell
def _(df_tratado):
    df_valor_unico = df_tratado.unique()
    return (df_valor_unico,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analisando os checando os dados lines
    """)
    return


@app.cell
def _(df_valor_unico):
    df_valor_unico.write_parquet('./data/tratado/wobs.parquet',)
    return


if __name__ == "__main__":
    app.run()
