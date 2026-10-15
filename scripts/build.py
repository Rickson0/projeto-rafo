#!/usr/bin/env python3
"""
Build script do RAFO Explorer.

Pipeline:
    1. Carrega WOBs limpos + metadados de livros
    2. Detecta tags RAFO e separa falantes Dragonsteel
    3. Pareia pergunta ↔ resposta via join_asof
    4. Limpa texto (regex de ruído)
    5. Treina NMF (tópicos TF-IDF) e BERTopic (tópicos semânticos)
    6. Persiste resultados em public/ (parquet + JSON) para o WASM
    7. Exporta app.py para HTML+WASM em _site/

Uso:
    uv run --extra build scripts/build.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import polars as pl
from bertopic import BERTopic
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ══════════════════════════════════════════════════════════════════════
# Caminhos
# ══════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "wobs_limpos.parquet"
PUBLIC = ROOT / "public"
SITE = ROOT / "_site"

# ══════════════════════════════════════════════════════════════════════
# Vocabulário Cosmere
# ══════════════════════════════════════════════════════════════════════

PERSONAGENS = {
    'kaladin', 'shallan', 'dalinar', 'adolin', 'kelsier', 'vin', 'elend',
    'sazed', 'hoid', 'wit', 'wayne', 'wax', 'marasi', 'raoden', 'sarene',
    'vasher', 'vivenna', 'susebron', 'lightsong', 'khriss', 'nazh',
    'szeth', 'lift', 'navani', 'jasnah', 'renarin', 'gavilar', 'taravangian',
    'moash', 'teft', 'rock', 'lopen', 'pattern', 'syl', 'spook', 'ham',
    'breeze', 'clubs', 'dockson', 'marsh', 'tensoon', 'straff', 'zane',
    'tress', 'yumi', 'nomad', 'sigzil', 'rira', 'gawx', 'paina',
    'retribution', 'raboniel', 'talenel', 'talenelat', 'taln', 'kaminah',
    'makibak', 'rahel', 'aux', 'cinder king', 'painter', 'nikaro', 'glorf',
    'charlie', 'sorceress', 'riina', 'twinsoul', 'frost', 'dlavil', 'ulaam',
    'medelantorius', 'dedelin', 'jezrien', 'nale', 'chanarach', 'battar',
    'kalak', 'shalash', 'ishar', 'vedel', 'pailiah',
}

PLANETAS = {
    'scadrial', 'roshar', 'sel', 'nalthis', 'taldain', 'threnody',
    'yolen', 'ashyn', 'braize', 'canticle', 'komashi', 'lumar',
    'first of the sun', 'drominad', 'shadesmar', 'urithiru', 'kholinar',
    'luthadel', 'elantris', 'hallandren', "t'telir", 'azir', 'jah keved',
    'alethkar', 'thaylenah', 'herdaz', 'aimia', 'shinovar', 'miral',
    'dhatri', 'utol', 'kite planet', 'ocean in the sky planet',
    'freefall planet', 'dragonrider planet', 'reliquary',
}

LIVROS = {
    'mistborn', 'elantris', 'warbreaker', 'stormlight', 'cosmere',
    'way of kings', 'words of radiance', 'oathbringer', 'rhythm of war',
    'wind and truth', 'final empire', 'well of ascension', 'hero of ages',
    'alloy of law', 'shadows of self', 'bands of mourning', 'lost metal',
    'edgedancer', 'dawnshard', 'tress', 'yumi', 'sunlit man',
    'sixth of the dusk', 'shadows for silence', 'emperors soul',
    'secret history', 'white sand', 'the fires of december',
    'isles of the emberdark', "hoid's travails", 'blightfall',
    'the sunlit man',
}

LORE = {
    'shard', 'adonalsium', 'investiture', 'allomancy', 'feruchemy',
    'hemalurgy', 'surgebinding', 'stormlight', 'spren', 'radiant',
    'honor', 'preservation', 'ruin', 'odium', 'cultivation', 'endowment',
    'autonomy', 'ambition', 'dominion', 'devotion', 'whimsy', 'valor',
    'invention', 'mercy', 'virtuosity', 'reason', 'voidbringer',
    'parshmen', 'singers', 'listener', 'mistborn', 'coinshot',
    'lurcher', 'thug', 'soother', 'rioter', 'seeker', 'smoker',
    'bondsmith', 'windrunner', 'skybreaker', 'edgedancer', 'truthwatcher',
    'lightweaver', 'elsecaller', 'willshaper', 'stoneward', 'dustbringer',
    'compound', 'fullborn', 'retribution',
    'yoki-hijo', 'yoki hijo', 'hion', 'hion lines', 'shroud', 'nightmare',
    'nightmare painter',
    'aether', 'aetherbound', 'aether spores', 'luhel bond', 'sprouter',
    'midnight spores', 'midnight essence', 'verdant', 'roseite',
    'crimson', 'zephyr',
    'embers', 'ember men', 'ember women', 'chorus', 'charred',
    'grand knell', 'emberdark', 'unsea', 'intensifier',
    'negative investiture', 'awakened metalminds',
    'ati', 'leras', 'tanavast', 'koravellium avast', 'edgli', 'bavadin',
    'shattering', 'vessel', 'intent', 'dawnshards', 'change', 'survive',
    'unite', 'divide',
    'microkinesis', 'knacks',
    'axon', 'axi', 'kinetic investiture', 'anti-investiture', 'surge',
    'gravitation', 'adhesion', 'cohesion', 'tension', 'progression',
    'illumination', 'transformation', 'transportation', 'division',
    'abrasion',
    'nightwatcher', 'stormfather', 'sibling',
    'nahel bond', 'cognitive realm', 'physical realm', 'spiritual realm',
}

DRAGONSTEEL_FUNCIONARIOS = [
    "Brandon Sanderson", "Peter Ahlstrom", "Isaac Stewart", "Dan Wells",
    "Karen Ahlstrom", "Adam Horne", "Emily Sanderson", "Kara Stewart",
    "Matt Hatch", "Kathy",
]

DRAGONSTEEL_LOWERCASE = [n.lower() for n in DRAGONSTEEL_FUNCIONARIOS]
VOCABULARIO = PERSONAGENS | PLANETAS | LIVROS | LORE

# ══════════════════════════════════════════════════════════════════════
# Metadados dos livros
# ══════════════════════════════════════════════════════════════════════

LIVROS_META = pl.DataFrame({
    "livro": [
        "elantris", "final empire", "well of ascension", "hero of ages",
        "warbreaker", "way of kings", "alloy of law", "emperors soul",
        "shadows for silence", "words of radiance", "sixth of the dusk",
        "shadows of self", "bands of mourning", "secret history",
        "arcanum unbounded", "white sand", "oathbringer", "dawnshard",
        "rhythm of war", "lost metal", "tress", "yumi", "sunlit man",
        "wind and truth", "isles of the emberdark", "the fires of december",
    ],
    "ano": [
        2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2014,
        2015, 2016, 2016, 2016, 2016, 2017, 2020, 2020, 2022, 2023, 2023,
        2023, 2024, 2025, 2026,
    ],
    "serie": [
        "elantris", "mistborn era 1", "mistborn era 1", "mistborn era 1",
        "warbreaker", "stormlight", "mistborn era 2", "standalone",
        "standalone", "stormlight", "standalone", "mistborn era 2",
        "mistborn era 2", "mistborn era 2", "coletanea", "white sand",
        "stormlight", "stormlight", "stormlight", "mistborn era 2",
        "hoid's travails", "hoid's travails", "hoid's travails",
        "stormlight", "hoid's travails", "hoid's travails",
    ],
    "era": [
        "pre-rafo", "pre-rafo", "pre-rafo", "pre-rafo", "pre-rafo",
        "pre-rafo", "pre-rafo", "pre-rafo", "pre-rafo",
        "era rafo", "era rafo", "era rafo", "era rafo", "era rafo",
        "era rafo", "era rafo", "era rafo",
        "late cosmere", "late cosmere", "late cosmere",
        "late cosmere", "late cosmere", "late cosmere",
        "late cosmere", "late cosmere", "late cosmere",
    ],
})

# ══════════════════════════════════════════════════════════════════════
# 1. Carregamento e preparação
# ══════════════════════════════════════════════════════════════════════

def carregar_wobs() -> pl.DataFrame:
    """Carrega WOBs e faz join com metadados de era (1:1 por ano)."""
    if not DATA.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {DATA}\n"
            f"Baixe o dataset ou ajuste o caminho em scripts/build.py"
        )

    mapa_era = LIVROS_META.select("ano", "era").unique(subset=["ano"])

    df = (
        pl.scan_parquet(DATA)
          .join(mapa_era.lazy(), left_on="adicionado_ano", right_on="ano", how="left")
          .collect()
    )
    print(f"  → {len(df):,} WOBs carregados")
    return df


def filtrar_rafos(df: pl.DataFrame) -> pl.DataFrame:
    """Filtra apenas registros que contêm alguma tag RAFO."""
    rafos = (
        df.select(pl.col("tags").explode().unique())
          .filter(pl.col("tags").str.contains("rafo"))
          .to_series()
          .to_list()
    )
    print(f"  → {len(rafos)} variantes de tag RAFO: {rafos}")

    expr_rafo = pl.any_horizontal(
        [pl.col("tags").list.contains(r) for r in rafos]
    ).fill_null(False)

    df_rafo = df.with_columns(
        seq=pl.int_range(pl.len()).over("id")
    ).filter(expr_rafo)

    print(f"  → {len(df_rafo):,} registros com RAFO")
    return df_rafo


# ══════════════════════════════════════════════════════════════════════
# 2. Pareamento pergunta ↔ resposta
# ══════════════════════════════════════════════════════════════════════

def parear_perguntas_respostas(df_rafo: pl.DataFrame) -> pl.DataFrame:
    """Separa falantes Dragonsteel das perguntas e pareia cada pergunta
    com a próxima resposta do Brandon na mesma thread (via join_asof)."""
    padrao_dragonsteel = "|".join(re.escape(n) for n in DRAGONSTEEL_LOWERCASE)
    eh_dragonsteel = (
        pl.col("speaker")
          .str.to_lowercase()
          .str.contains(padrao_dragonsteel)
          .fill_null(False)
    )

    perguntas_seq = (
        df_rafo.filter(~eh_dragonsteel)
               .select(
                   "id", "seq",
                   pl.col("speaker").alias("speaker_p"),
                   pl.col("text").alias("pergunta"),
                   "date",
               )
    )
    respostas_seq = (
        df_rafo.filter(eh_dragonsteel)
               .select(
                   "id", "seq",
                   pl.col("speaker").alias("speaker_r"),
                   pl.col("text").alias("resposta"),
                   "date",
               )
    )

    dataset = (
        perguntas_seq
        .sort("id", "seq")
        .join_asof(
            respostas_seq.sort("id", "seq"),
            on="seq", by="id", strategy="forward",
        )
        .drop("seq")
        .filter(pl.col("resposta").is_not_null())
    )

    print(f"  → {len(dataset):,} pares pergunta↔resposta")
    return dataset


# ══════════════════════════════════════════════════════════════════════
# 3. Limpeza de texto
# ══════════════════════════════════════════════════════════════════════

REGEX_RUIDO = (
    r"(?i)\b(okay|sighs|sighing|laughter|applause|yeah|little|know|like|"
    r"um|uh|inaudible|did|does|question|story|gonna|do|thats|right|rafo|"
    r"seen|going|look|new|just|the|of|that|to|is|in|it|and|are|you)\b"
)


def _limpar(col: str) -> pl.Expr:
    return (
        pl.col(col)
          .str.to_lowercase()
          .str.replace_all(r"[\(\[\{].*?[\)\]\}]", "")
          .str.replace_all(r"[^\w\s]", "")
          .str.replace_all(REGEX_RUIDO, "")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
    )


def limpar_dataset(dataset: pl.DataFrame) -> pl.DataFrame:
    """Aplica limpeza de texto e cria coluna combinada `text_full`."""
    df_limpo = dataset.with_columns(
        _limpar("pergunta").alias("pergunta_limpa"),
        _limpar("resposta").alias("resposta_limpa"),
    )

    df_text_full = df_limpo.with_columns(
        text_full=pl.col("pergunta_limpa") + " " + pl.col("resposta_limpa"),
        ano=pl.col("date").dt.year(),
    )
    print(f"  → {len(df_text_full):,} documentos após limpeza")
    return df_text_full


# ══════════════════════════════════════════════════════════════════════
# 4. NMF (tópicos por TF-IDF)
# ══════════════════════════════════════════════════════════════════════

def rodar_nmf(df_limpo: pl.DataFrame, n_topicos: int = 8) -> pl.DataFrame:
    """Roda NMF sobre perguntas limpas e retorna tabela com top termos."""
    vetorizador = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.7,
        min_df=2,
    )
    X = vetorizador.fit_transform(df_limpo["pergunta_limpa"].to_list())

    nmf = NMF(n_components=n_topicos, max_iter=1000, random_state=42)
    nmf.fit(X)

    termos = vetorizador.get_feature_names_out()
    linhas = []
    for i, comp in enumerate(nmf.components_):
        top = comp.argsort()[:-6:-1]
        linhas.append({
            "topico": i + 1,
            "termos": ", ".join(termos[j] for j in top),
        })

    resultado = pl.DataFrame(linhas)
    print(f"  → NMF com {n_topicos} tópicos treinado")
    return resultado


# ══════════════════════════════════════════════════════════════════════
# 5. BERTopic (tópicos semânticos)
# ══════════════════════════════════════════════════════════════════════

def rodar_bertopic(
    df_text_full: pl.DataFrame,
    n_topicos: int = 8,
) -> BERTopic:
    """Treina BERTopic com vocabulário Cosmere controlado."""
    vectorizer_model = CountVectorizer(vocabulary=VOCABULARIO)

    topic_model = BERTopic(
        language="english",
        calculate_probabilities=True,
        verbose=True,
        vectorizer_model=vectorizer_model,
        nr_topics=n_topicos,
    )

    textos = df_text_full["text_full"].to_list()
    topics, probs = topic_model.fit_transform(textos)

    print(f"  → BERTopic: {len(set(topics))} tópicos encontrados")
    return topic_model


# ══════════════════════════════════════════════════════════════════════
# 6. Persistência dos resultados
# ══════════════════════════════════════════════════════════════════════

def _pandas_para_polars(obj) -> pl.DataFrame:
    """Converte pandas.DataFrame → polars.DataFrame (compat com BERTopic)."""
    if isinstance(obj, pl.DataFrame):
        return obj
    # pandas ou qualquer coisa que tenha .to_dict(orient="records")
    try:
        return pl.from_pandas(obj)
    except Exception:
        return pl.DataFrame(obj)


def salvar_resultados(
    topic_model: BERTopic,
    df_text_full: pl.DataFrame,
    nmf_result: pl.DataFrame,
) -> None:
    """Salva tudo que o app WASM precisa consumir em public/ (parquet + JSON)."""
    PUBLIC.mkdir(parents=True, exist_ok=True)

    # ── Topic info (BERTopic devolve pandas) ──────────────────────────
    topic_info = _pandas_para_polars(topic_model.get_topic_info())
    topic_info.write_parquet(PUBLIC / "topic_info.parquet")
    print(f"  → public/topic_info.parquet ({len(topic_info)} linhas)")

    # ── Tópicos ao longo do tempo (também pandas) ─────────────────────
    topics_ot = _pandas_para_polars(
        topic_model.topics_over_time(
            df_text_full["text_full"].to_list(),
            timestamps=df_text_full["ano"].to_list(),
            nr_bins=10,
        )
    )
    topics_ot.write_parquet(PUBLIC / "topics_over_time.parquet")
    print(f"  → public/topics_over_time.parquet ({len(topics_ot)} linhas)")

    # ── Atribuição doc → tópico ───────────────────────────────────────
    doc_topico = pl.DataFrame({
        "topico": topic_model.topics_,
        "ano": df_text_full["ano"].to_list(),
    })
    doc_topico.write_parquet(PUBLIC / "doc_topicos.parquet")
    print(f"  → public/doc_topicos.parquet ({len(doc_topico)} linhas)")

    # ── Top palavras por tópico (barchart) ────────────────────────────
    linhas_palavras = []
    for topico_id in topic_model.get_topics():
        if topico_id == -1:
            continue
        for palavra, peso in topic_model.get_topic(topico_id)[:10]:
            linhas_palavras.append({
                "topico": int(topico_id),
                "palavra": str(palavra),
                "peso": float(peso),
            })
    topic_palavras = pl.DataFrame(linhas_palavras)
    topic_palavras.write_parquet(PUBLIC / "topic_palavras.parquet")
    print(f"  → public/topic_palavras.parquet ({len(topic_palavras)} linhas)")

    # ── Hierarquia (opcional) ─────────────────────────────────────────
    try:
        hierarchy = topic_model.hierarchy_topics_()
        (PUBLIC / "hierarchy.json").write_text(json.dumps(hierarchy, default=str))
        print("  → public/hierarchy.json")
    except Exception as e:
        print(f"  ⚠ Hierarquia não gerada: {e}")

    # ── NMF (já é polars) ─────────────────────────────────────────────
    nmf_result.write_parquet(PUBLIC / "nmf_topicos.parquet")
    print(f"  → public/nmf_topicos.parquet ({len(nmf_result)} linhas)")

    # ── Dataset completo (polars) ─────────────────────────────────────
    dataset_export = df_text_full.select(
        "date", "speaker_p", "speaker_r", "pergunta", "resposta",
        "text_full", "ano",
    )
    dataset_export.write_parquet(PUBLIC / "dataset.parquet")
    print(f"  → public/dataset.parquet ({len(dataset_export)} linhas)")

    # ═════════════════════════════════════════════════════════════════
    # Versões JSON para o WASM (evita dependência do pyarrow no browser)
    # ═════════════════════════════════════════════════════════════════
    def _para_json(df: pl.DataFrame, nome: str) -> None:
        """Serializa para JSON. Polars lida com Datetime e List automaticamente."""
        (PUBLIC / f"{nome}.json").write_text(
            json.dumps(df.to_dicts(), default=str)
        )
        print(f"  → public/{nome}.json")

    _para_json(topic_info,     "topic_info")
    _para_json(topic_palavras, "topic_palavras")
    _para_json(topics_ot,      "topics_over_time")
    _para_json(doc_topico,     "doc_topicos")
    _para_json(nmf_result,     "nmf_topicos")
    _para_json(dataset_export, "dataset")


# ══════════════════════════════════════════════════════════════════════
# 7. Export para WASM
# ══════════════════════════════════════════════════════════════════════

def exportar_wasm() -> None:
    """Exporta app.py para HTML+WASM e copia assets estáticos."""
    SITE.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "marimo", "export", "html-wasm",
        "app.py",
        "--mode", "run",
        "--no-show-code",
        "-o", str(SITE / "index.html"),
    ], check=True, cwd=ROOT)

    # Copiar public/ para _site/public/
    if PUBLIC.exists():
        shutil.copytree(PUBLIC, SITE / "public", dirs_exist_ok=True)

    # .nojekyll evita processamento pelo Jekyll no GitHub Pages
    (SITE / ".nojekyll").touch()

    print(f"  → _site/index.html + _site/public/ ({len(list(PUBLIC.iterdir()))} assets)")


# ══════════════════════════════════════════════════════════════════════
# Orquestração
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("═" * 60)
    print("🔨 RAFO Explorer — build")
    print("═" * 60)

    print("\n[1/7] Carregando WOBs…")
    df = carregar_wobs()

    print("\n[2/7] Filtrando RAFOs…")
    df_rafo = filtrar_rafos(df)

    print("\n[3/7] Pareando perguntas ↔ respostas…")
    dataset = parear_perguntas_respostas(df_rafo)

    print("\n[4/7] Limpando texto…")
    df_text_full = limpar_dataset(dataset)

    df_limpo = df_text_full.select("pergunta_limpa", "resposta_limpa")

    print("\n[5/7] Rodando NMF…")
    nmf_result = rodar_nmf(df_limpo, n_topicos=8)

    print("\n[6/7] Rodando BERTopic (pode demorar ~1–2min)…")
    topic_model = rodar_bertopic(df_text_full, n_topicos=8)

    print("\n[7/7] Salvando resultados e exportando WASM…")
    salvar_resultados(topic_model, df_text_full, nmf_result)
    exportar_wasm()

    print("\n" + "═" * 60)
    print("🎉 Build completo!")
    print(f"   Teste local:  python -m http.server -d {SITE} 8000")
    print("   Abra:         http://localhost:8000")
    print("═" * 60)


if __name__ == "__main__":
    main()