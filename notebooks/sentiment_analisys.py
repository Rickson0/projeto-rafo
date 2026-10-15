import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import polars as pl
    import polars.selectors as cs
    from textblob import TextBlob
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import nltk
    from nltk.corpus import stopwords
    import re
    from collections import Counter
    import seaborn as sns
    nltk.download('stopwords')
    return (
        Counter,
        SentimentIntensityAnalyzer,
        TextBlob,
        WordCloud,
        mo,
        nltk,
        pl,
        plt,
        re,
        stopwords,
    )


@app.cell
def _():
    # Personagens
    personagens = {
        'kaladin', 'shallan', 'dalinar', 'adolin', 'kelsier', 'vin', 'elend',
        'sazed', 'hoid', 'wit', 'wayne', 'wax', 'marasi', 'raoden', 'sarene',
        'vasher', 'vivenna', 'susebron', 'lightsong', 'khriss', 'nazh',
        'szeth', 'lift', 'navani', 'jasnah', 'renarin', 'gavilar', 'taravangian',
        'moash', 'teft', 'rock', 'lopen', 'pattern', 'syl', 'spook', 'ham',
        'breeze', 'clubs', 'dockson', 'marsh', 'tensoon', 'straff', 'zane',
        'tress', 'yumi', 'nomad', 'sigzil', 'rira', 'gawx', 'paina'
    }

    # Planetas e locais
    planetas = {
        'scadrial', 'roshar', 'sel', 'nalthis', 'taldain', 'threnody',
        'yolen', 'ashyn', 'braize', 'canticle', 'komashi', 'lumar',
        'first of the sun', 'drominad', 'shadesmar', 'urithiru', 'kholinar',
        'luthadel', 'elantris', 'hallandren', 't' 'telir', 'azir', 'jah keved',
        'alethkar', 'thaylenah', 'herdaz', 'aimia', 'shinovar','miral'
    }

    # Livros e séries
    livros = {
        'mistborn', 'elantris', 'warbreaker', 'stormlight', 'cosmere',
        'way of kings', 'words of radiance', 'oathbringer', 'rhythm of war',
        'wind and truth', 'final empire', 'well of ascension', 'hero of ages',
        'alloy of law', 'shadows of self', 'bands of mourning', 'lost metal',
        'edgedancer', 'dawnshard', 'tress', 'yumi', 'sunlit man',
        'sixth of the dusk', 'shadows for silence', 'emperors soul',
        'secret history', 'white sand'
    }

    # Termos do lore
    lore = {
        'shard', 'adonalsium', 'investiture', 'allomancy', 'feruchemy',
        'hemalurgy', 'surgebinding', 'stormlight', 'spren', 'radiant',
        'honor', 'preservation', 'ruin', 'odium', 'cultivation', 'endowment',
        'autonomy', 'ambition', 'dominion', 'devotion', 'whimsy', 'valor',
        'invention', 'mercy', 'virtuosity', 'reason', 'voidbringer',
        'parshmen', 'singers', 'listener', 'mistborn', 'coinshot',
        'lurcher', 'thug', 'soother', 'rioter', 'seeker', 'smoker',
        'bondsmith', 'windrunner', 'skybreaker', 'edgedancer', 'truthwatcher',
        'lightweaver', 'elsecaller', 'willshaper', 'stoneward', 'dustbringer'
    }

    # Expressões-chave que você quer manter
    expressoes = {
        'rafo', "i'll", 'brilliant', 'question', 'read', 'find', 'out',
        'journey', 'before', 'destination', 'life', 'before', 'death',
        'strength', 'before', 'weakness', 'journey before destination'
    }

    dragonsteel_funcionarios = [
        "Brandon Sanderson",      # Fundador e principal respondedor
        "Peter Ahlstrom",         # VP de Editorial
        "Isaac Stewart",          # VP de Desenvolvimento Criativo
        "Dan Wells",              # VP de Narrativa
        "Karen Ahlstrom",         # Editora de Continuidade
        "Adam Horne",             # VP de Publicidade e Marketing
        "Emily Sanderson",        # COO (Chief Operating Officer)
        "Kara Stewart",           # CFO (Chief Financial Officer)
        "Matt Hatch",             # VP de Operações
        "Kathy",                  # Assistente da Emily (aparece em alguns registros)
    ]
    dragonsteel_lowercase = [x.lower() for x in dragonsteel_funcionarios]

    vocabulario = personagens | planetas | livros | lore | expressoes
    padrao_rafo = r'(?i)\brafo\w*\b|read\s+and\s+find\s+out'
    return dragonsteel_lowercase, lore, padrao_rafo, vocabulario


@app.cell
def _(SentimentIntensityAnalyzer, nltk):
    nltk.download('vader_lexicon', quiet=True)
    analyzer = SentimentIntensityAnalyzer()

    # ============================================================
    # 1. VARIAÇÕES DE RAFO
    # ============================================================


    rafo_variations = {
        # --- Forma padrão ---
        'rafo': 0.0,
        'RAFO': 0.0,
        'Rafo': 0.0,

        # --- Formas verbais (RAFO'd, RAFOed, etc.) ---
        "rafo'd": -0.3,      # "fui RAFO'd" = leve frustração
        "RAFO'd": -0.3,
        "Rafo'd": -0.3,
        'rafoed': -0.3,
        'RAFOed': -0.3,
        'Rafoed': -0.3,
        "rafo'ed": -0.3,
        "RAFO'ed": -0.3,
        'rafoing': 0.0,
        'RAFOing': 0.0,
        'Rafoing': 0.0,
        "rafo'ing": 0.0,
        "RAFO'ing": 0.0,
        'rafos': 0.0,
        'RAFOs': 0.0,
        'Rafos': 0.0,

        # --- Frase completa ---
        'read and find out': 0.0,
        'Read and Find Out': 0.0,
        'Read And Find Out': 0.0,
        'READ AND FIND OUT': 0.0,
        'read & find out': 0.0,
        'Read & Find Out': 0.0,

        # --- Com pontos ---
        'r.a.f.o.': 0.0,
        'R.A.F.O.': 0.0,

        # --- Espaçado ---
        'r a f o': 0.0,
        'R A F O': 0.0,
    }

    # ============================================================
    # 2. MARCADORES DE "BOA PERGUNTA"
    # ============================================================
    # Quando Brandon diz "good question" antes de um RAFO,
    # é um RAFO positivo (a pergunta foi tão boa que ele não pode responder)

    good_question_markers = {
        'good question': 1.5,
        'great question': 2.0,
        'excellent question': 2.0,
        'interesting question': 1.5,
        'really good question': 2.0,
        'very good question': 2.0,
        'fantastic question': 2.5,
        'wonderful question': 2.0,
        'amazing question': 2.5,
        'perfect question': 2.5,
        "that's a good question": 1.5,
        'that is a good question': 1.5,
        "that's a great question": 2.0,
        'that is a great question': 2.0,
        "that's an excellent question": 2.0,
        "that's an interesting question": 1.5,
        "that's a really good question": 2.0,
        "that's a very good question": 2.0,
        'good question, rafo': 1.0,      # pergunta boa + RAFO
        'great question, rafo': 1.5,
        'excellent question, rafo': 1.5,
        'interesting question, rafo': 1.0,
        "that's a good question, rafo": 1.0,
        "that's a great question, rafo": 1.5,
    
    }

    # ============================================================
    # 3. FRASES DE CONTEXTO RAFO
    # ============================================================

    rafo_context = {
        # --- RAFO simples ---
        "that's a rafo": 0.0,
        "that is a rafo": 0.0,
        "it's a rafo": 0.0,
        "this is a rafo": 0.0,

        # --- RAFO grande (frustrante pra fã) ---
        "that's a big rafo": -0.5,
        "that's a big big rafo": -1.0,
        "that's a huge rafo": -0.8,
        "that's a massive rafo": -1.0,
        "big rafo": -0.5,
        "huge rafo": -0.8,

        # --- Você foi RAFO'd ---
        "you've been rafo'd": -0.5,
        "you have been rafo'd": -0.5,
        "you got rafo'd": -0.5,
        "i've been rafo'd": -0.5,
        "we've been rafo'd": -0.5,

        # --- Vou te dar um RAFO ---
        "i'll rafo that": 0.0,
        "i will rafo that": 0.0,
        "i'm going to rafo that": 0.0,
        "i have to rafo that": 0.0,
        "i'm rafoing that": 0.0,

        # --- RAFO com desculpa ---
        "rafo, i'm afraid": -0.5,
        "rafo i'm afraid": -0.5,
        "i'm afraid that's a rafo": -0.5,
        "sorry, rafo": -0.5,
        "sorry rafo": -0.5,

        # --- RAFO definitivo ---
        "definite rafo": -0.3,
        "absolute rafo": -0.3,
        "total rafo": -0.3,
    }

    # ============================================================
    # 4. EMOJIS E EMOTICONS
    # ============================================================

    emojis = {
        # --- Smileys positivos ---
        ':)': 2.0, ':-)': 2.0, '=)': 2.0,
        ';)': 1.5, ';-)': 1.5,
        ':D': 2.5, ':-D': 2.5,
        ':]': 1.5, ':-]': 1.5,
    

        # --- Smileys negativos ---
        ':(': -2.0, ':-(': -2.0,
        ':/': -1.0, ':-/': -1.0,
        ':|': 0.0, ':-|': 0.0,
          # palhaço = vergonha alheia

        # --- Outros ---
 
    }

    # ============================================================
    # 5. TERMOS DO COSMERE (neutros)
    # ============================================================

    # sanderson_terms = {
    #     'cosmere': 0.0, 'shard': 0.0, 'shardblade': 1.5,
    #     'investiture': 0.5, 'allomancy': 1.0, 'feruchemy': 1.0,
    #     'hemalurgy': -2.0, 'surgebinding': 1.0,
    #     'spren': 0.5, 'highstorm': 0.0, 'everstorm': -2.0,
    #     'desolation': -3.0, 'odium': -3.5, 'honor': 2.5,
    #     'ruin': -3.0, 'preservation': 2.0, 'cultivation': 1.0,
    #     'endowment': 1.0, 'devotion': 2.0, 'dominion': 0.0,
    #     'autonomy': 0.0, 'radiant': 2.0, 'voidbringer': -2.5,
    #     'mistborn': 1.0, 'koloss': -1.5, 'storming': -1.0,
    #     'storms': -1.0, 'bridgeboy': 0.0, 'nicrosil': 0.0,
    #     'wob': 0.0, 'words of brandon': 0.0,
    # }



    all_lexicon = {}
    all_lexicon.update(rafo_variations)
    all_lexicon.update(good_question_markers)
    all_lexicon.update(rafo_context)
    all_lexicon.update(emojis)


    analyzer.lexicon.update(all_lexicon)
    return (analyzer,)


@app.cell
def _(dragonsteel_lowercase, pl):
    lf = (
        pl.scan_parquet('data/processed/wobs_limpos.parquet')
        .with_columns(
            pl.when(pl.col('speaker').str.to_lowercase().is_in(dragonsteel_lowercase))
              .then(pl.lit('Respondedor'))
              .otherwise(pl.lit('Questionador'))
              .alias('papel')
        )
        .with_columns(
            (pl.col('papel') == 'Questionador').cum_sum().over('id').alias('turno')
        )
        .with_columns(
            pl.when(pl.col('turno') == 0).then(1).otherwise(pl.col('turno')).alias('turno')
        )
        .with_columns(
            pl.when(pl.col('papel') == 'Questionador')
              .then(pl.lit(0, dtype=pl.Int8))
              .otherwise(pl.lit(1, dtype=pl.Int8))
              .alias('_ordem')
        )
        # Ordem final: id → turno → papel (Q antes de R) → posição original
        .sort(
            ['id', 'turno', '_ordem', pl.int_range(pl.len()).over('id')]
        )
        .drop('_ordem')
    )

    df = lf.collect()
    return (df,)


@app.cell
def _(analyzer, df, pl):
    analise_sentimento = df.with_columns(
        sentimento = 
            pl.col('text')
                .map_elements(lambda t: analyzer.polarity_scores(t)['compound']
                              ,return_dtype=pl.Float64
            )
    )



    media_sentimento = analise_sentimento.group_by('id').agg(
        pl.col('sentimento').mean().alias('media_sentimento')
    ).sort('media_sentimento', descending=True)
    return (analise_sentimento,)


@app.cell
def _(analise_sentimento, pl):
    valor_sentimentos = analise_sentimento.group_by('id').agg(valor_sentimento = pl.col('sentimento').sum()
    ).sort('valor_sentimento', descending=True)
    valor_sentimentos
    return


@app.cell
def _(analise_sentimento):
    analise_sentimento['speaker'].unique()
    return


@app.cell
def _(analise_sentimento, pl):
    analise_sentimento.filter(pl.col('speaker') == 'Brandon Sanderson')
    return


@app.cell
def _(TextBlob, df, pl):

    analise_sentimento_text_blob = df.with_columns(
        sentimento = pl.col('text').map_elements(lambda t: TextBlob(t).polarity, return_dtype=pl.Float64),
        subjetividade = pl.col('text').map_elements(lambda t: TextBlob(t).subjectivity, return_dtype=pl.Float64)

    )
    return (analise_sentimento_text_blob,)


@app.cell
def _(analise_sentimento_text_blob):
    analise_sentimento_text_blob
    return


@app.cell
def _(analise_sentimento_text_blob, pl):
    analise_sentimento_text_blob.filter(pl.col('speaker') == 'Brandon Sanderson')
    return


@app.cell
def _(WordCloud, df, pl, plt, stopwords):
    extras = {
        'like', 'would', 'could', 'should', 'really', 'just', 'get', 'got',
         'also', 'even', 'much', 'many', 'still', 'back', 'going',
        'go', 'went', 'make', 'made', 'see', 'saw', 'know', 'think', 'thought',
        'want', 'need', 'use', 'used', 'using', 'thing', 'things', 'way',
        'time', 'times', 'day', 'days', 'year', 'years', 'people', 'im',
        'ive', 'id', 'ill', 'dont', 'didnt', 'cant', 'wont', 'youre',
        'thats', 'its', 'hes', 'shes', 'theyre', 'weve', 'youve'
    }
    stopwords_en = set(stopwords.words('english'))

    # stopwords_en.update(extras)
    texto = ''.join(df.select(pl.col('text')
        .cast(pl.Utf8))
        .to_series()
        .to_list())
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords_en,
        max_words=50,
        collocations=False,
        random_state=42
    ).generate(texto)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return stopwords_en, texto, wc


@app.cell
def _(WordCloud, df, pl, plt, stopwords_en, wc):
    df_sanderson = df.filter(pl.col('speaker') == 'Brandon Sanderson')
    texto_sanderson = ''.join(df_sanderson['text'])

    wc_sanderson = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords_en,
        max_words=50,
        collocations=False,
        random_state=42
    ).generate(texto_sanderson)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return


@app.cell
def _(WordCloud, df, pl, plt, stopwords_en, wc):
    df_rafo = df.filter(pl.col('text').str.contains('(?i)RAFO'))
    rafo_texto = ''.join(df_rafo['text'])
    word_cloud_rafo = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords_en,
        max_words=50,
        collocations=False,
        random_state=42
    ).generate(rafo_texto)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return


@app.cell
def _():
    # df_rafo_sanderson = df_sanderson.filter(pl.col('text').str.contains('(?i)RAFO'))
    # rafo_texto_sanderson = ''.join(df_rafo['text'])
    # word_cloud_rafo_sanderson = WordCloud(
    #     width=800,
    #     height=400,
    #     background_color='white',
    #     stopwords=stopwords_en,
    #     max_words=50,
    #     collocations=False,
    #     random_state=42
    # ).generate(rafo_texto_sanderson)

    # plt.figure(figsize=(12, 6))
    # plt.imshow(wc, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()
    return


@app.cell
def _(Counter, WordCloud, df, pl, plt, re, texto, vocabulario, wc):
    texto_cosmere = ' '.join(
        df.select(pl.col('text').cast(pl.Utf8))
          .to_series()
          .to_list()
    )

    tokens = re.findall(r"[a-z']+", texto.lower())

    # 4. Filtrar pelo vocabulário curado
    tokens_filtrados = [t for t in tokens if t in vocabulario]

    # 5. Contar frequências
    freq = Counter(tokens_filtrados)
    wc_cosmere = WordCloud(
        width=1200,
        height=700,
        background_color='white',
        colormap='viridis',
        max_words=150,
        collocations=False,
        random_state=42
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Cosmere Word Cloud', fontsize=18)
    plt.show()

    wc_cosmere.to_file('cosmere_wordcloud.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    Hipótese: RAFO correlaciona com sentimento positivo?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A resposta é não, ou vader é louco. Provalvemente é a segunda opção
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(analise_sentimento, padrao_rafo, pl):
    df_tem_rafo = analise_sentimento.with_columns(
        tem_rafo = (pl.col('text').str.contains(padrao_rafo)) 
    )
    df_tem_rafo
    return


@app.cell
def _(analise_sentimento, pl):
    perguntas_sobre_kelsier = analise_sentimento.filter(
        (pl.col('text').str.contains('(?i)kelsier'))
    ).select(
        pl.col('id').unique()
    )

    resultado_kelsier = analise_sentimento.join(perguntas_sobre_kelsier, on='id', how='semi')
    resultado_kelsier[['id','date','speaker', 'text', 'sentimento']]
    return


@app.cell
def _(analise_sentimento, pl):
    aparicoes_de_retribution = analise_sentimento.filter(
        pl.col('text').str.contains('(?i)retribution')   
    ).select(
        pl.col('id').unique()   
    )


    resultado_retribution = analise_sentimento.join(aparicoes_de_retribution, on='id', how='semi')
    resultado_retribution
    return


@app.cell
def _(analise_sentimento, pl):
    identity = analise_sentimento.filter(
        pl.col('text').str.contains('Investiture|Intent|Connection|Identity', )
    ).select(
        pl.col('id').unique()
    )
    resultado_identity = analise_sentimento.join(identity, on='id', how='semi')
    resultado_identity
    return


@app.cell
def _(analise_sentimento, lore, pl):
    padrao = '|'.join(lore)
    rafo= analise_sentimento.join(analise_sentimento.filter(pl.col('text').str.contains('(?i)RAFO') & pl.col('text').str.to_lowercase().str.contains(padrao) ).select(
        pl.col('id').unique()), on='id', how='semi')

    rafo
    return (rafo,)


@app.cell
def _(WordCloud, plt, rafo, stopwords_en, wc):
    rafo_texto_sanderson = ''.join(rafo['text'])
    wc_rafo = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords_en,
        max_words=150,
        collocations=False,
        random_state=42
    ).generate(rafo_texto_sanderson)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return


@app.cell
def _(analise_sentimento, pl):

    analise_sentimento.join(analise_sentimento.filter(pl.col('tags').list.contains('rafo-plus')).select(pl.col('id').unique()), on='id', how='semi')
    return


@app.cell
def _(analise_sentimento):
    analise_sentimento
    return


@app.cell
def _(analise_sentimento, pl):
    analise_sentimento.group_by('date').agg(
        pl.col('')
    )
    return


if __name__ == "__main__":
    app.run()
