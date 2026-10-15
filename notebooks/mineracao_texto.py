import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import json
    import unicodedata
    import re
    from bs4 import BeautifulSoup

    return BeautifulSoup, pl


@app.cell
def _(BeautifulSoup):

    def extrair_texto(html: str) -> str:
        if not html:
            return ""
        return BeautifulSoup(html, "html.parser").get_text(separator=" ")



    return (extrair_texto,)


@app.cell
def _(extrair_texto, pl):
    lf = pl.scan_parquet('data/tratado/wobs.parquet')

    df = (
        lf.with_columns(
            pl.col('speaker').map_elements(extrair_texto,return_dtype=pl.String),
            pl.col('text').map_elements(extrair_texto, return_dtype=pl.String),
            pl.col('note')
        ).sort('id')
    ).collect(engine='auto')

    return (df,)


@app.cell
def _():
    return


@app.cell
def _(df, pl):
    df_data_padronizada = df.with_columns(
        pl.col('event_date','date').str.to_date(),
        pl.col('modified_date').str.to_datetime(time_zone='UTC')
    ).with_columns(
        evento_ano = pl.col('event_date').dt.year(),
        adicionado_ano = pl.col('date').dt.year(),
        modificado_ano = pl.col('modified_date').dt.year()
    )
    df_data_padronizada
    return (df_data_padronizada,)


@app.cell
def _(df_data_padronizada):
    df_data_padronizada.write_parquet('data/processed/wobs_limpos.parquet')
    return


if __name__ == "__main__":
    app.run()
