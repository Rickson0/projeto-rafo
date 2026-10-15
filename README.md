# 📚 RAFO Explorer

> Análise interativa das perguntas que o **Brandon Sanderson** responde com
> **RAFO** (_Read And Find Out_) em Q&As e sessões de autógrafos.

[![Deploy](https://github.com/Rickson0/projeto-rafo/actions/workflows/deploy.yml/badge.svg)](https://github.com/Rickson0/projeto-rafo/actions/workflows/deploy.yml)
[![Built with marimo](https://img.shields.io/badge/built%20with-marimo-blue)](https://marimo.io)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org)

**🌐 Demo ao vivo:** https://Rickson0.github.io/projeto-rafo/

---

## 🎯 Sobre o projeto

RAFO é o termo que o Sanderson usa em Q&As para dizer "leia e descubra" — uma
não-resposta educada que tanto pode indicar spoiler quanto uma pergunta boa
demais para ser respondida de imediato. Ao longo de quase 20 anos, ele já
distribuiu **milhares** dessas respostas.

Este projeto explora esse corpus com **modelagem de tópicos** (NMF + BERTopic)
para responder perguntas como:

- 🧩 **Que tipo de pergunta gera RAFO?** (personagens, planetas, magia, lore…)
- 📅 **Como os RAFOs mudaram ao longo do tempo?** Antes/depois do Cosmere?
- 🔥 **Qual o RAFO mais antigo ainda não resolvido?**
- 📊 **Qual a frequência de RAFOs por ano?**
- ⚛️ **Como os RAFOs se relacionam com Realmatics?** (Connection, Intent,
  Identity, Investiture, os três Reinos…)

---

## ✨ Features

- 🎛️ **Dashboard interativo** — filtros reativos por ano, tópico e busca textual
- 🧠 **Dois modelos de tópicos** rodando em paralelo:
  - **NMF** sobre TF-IDF (interpretabilidade clássica)
  - **BERTopic** com vocabulário Cosmere controlado (tópicos semânticos)
- 📈 **Visualizações interativas** em Altair
- 🚀 **Zero backend** — roda 100% no navegador via WASM (Pyodide)
- 🔄 **Deploy automático** via GitHub Actions

---

## 🚀 Como usar

### 🌐 Versão online

Acesse **[Rickson0.github.io/projeto-rafo](https://Rickson0.github.io/projeto-rafo/)**
e interaja direto no navegador — sem instalação.

### 💻 Localmente (modo notebook)

Para editar/explorar o notebook reativo:

```bash
git clone https://github.com/Rickson0/projeto-rafo.git
cd projeto-rafo
uv sync --extra build
uv run marimo edit app.py
```

O marimo abre uma interface no navegador onde cada célula reage às outras
em tempo real — diferente de Jupyter, onde você executa manualmente.

### 🔨 Rodar o build local

Para gerar o site estático (WASM) igual ao que o CI produz:

```bash
uv run --extra build scripts/build.py
python -m http.server -d _site 8000
# Abra http://localhost:8000
```

---

## 🏗️ Arquitetura

```
projeto-rafo/
├── app.py                     # Notebook marimo (versão WASM)
├── scripts/
│   └── build.py               # Pipeline: treina modelos + exporta WASM
├── data/
│   └── processed/
│       └── wobs_limpos.parquet  # Corpus de WOBs (3.5 MB)
├── public/                    # Assets pré-computados (gerados no CI)
│   ├── topic_info.parquet
│   ├── topic_palavras.parquet
│   ├── topics_over_time.parquet
│   ├── doc_topicos.parquet
│   ├── nmf_topicos.parquet
│   └── dataset.parquet
├── _site/                     # Site estático final (gerado)
├── .github/workflows/
│   └── deploy.yml             # CI: build → GitHub Pages
├── pyproject.toml
└── README.md
```

### 🧠 Por que dois artefatos (`public/` e `_site/`)?

O **BERTopic** e o **scikit-learn** não rodam no Pyodide (WASM). Então a
estratégia é **pré-computar** os resultados pesados no CI e commitar apenas
os parquets resultantes em `public/`. O `app.py` no navegador só lê esses
arquivos e reconstrói as visualizações com Altair — leve e rápido.

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│ Corpus WOBs │ ──▶ │  build.py    │ ──▶ │  public/   │
│ (parquet)   │     │ NMF+BERTopic │     │ (parquets) │
└─────────────┘     └──────────────┘     └────────────┘
                                                │
                                                ▼
                     ┌──────────────┐     ┌────────────┐
                     │  app.py      │ ◀── │  WASM      │
                     │  (marimo)    │     │ (Pyodide)  │
                     └──────────────┘     └────────────┘
                                                │
                                                ▼
                                         GitHub Pages
```

---

## 🛠️ Stack

| Camada | Ferramenta | Por quê |
|---|---|---|
| Notebook reativo | [**marimo**](https://marimo.io) | Reatividade real, sem estado escondido |
| Dataframes | [**Polars**](https://pola.rs) | Mais rápido que pandas, Lazy API |
| Topic modeling | [**BERTopic**](https://maartengr.github.io/BERTopic/) + [**scikit-learn**](https://scikit-learn.org) | NMF + embeddings |
| Visualização | [**Altair**](https://altair-viz.github.io) | Declarativo, funciona no WASM |
| Deploy | **GitHub Pages** + **Pyodide** | Zero backend, zero custo |
| Gerenciamento | [**uv**](https://github.com/astral-sh/uv) | Instalação em segundos |

---

## 📊 Pipeline de dados

1. **Carregamento** — WOBs limpos + metadados de livros (ano, série, era)
2. **Detecção de RAFO** — filtra registros com tags `rafo*`
3. **Pareamento** — `join_asof` pareia cada pergunta do fã com a próxima
   resposta do Sanderson na mesma thread
4. **Limpeza** — remove ruído de transcrição (`um`, `uh`, `[laughter]`, etc.)
5. **NMF** — TF-IDF + decomposição em ~8 tópicos
6. **BERTopic** — embeddings + vocabulário Cosmere controlado (~500 termos)
7. **Export** — parquets em `public/` + HTML/WASM em `_site/`

---

## 🤝 Contribuindo

Ideias de melhoria são bem-vindas! Algumas que estão na fila:

- [ ] Classificador "RAFO resolvido?" (detectar quando o Sanderson respondeu depois)
- [ ] Análise de Realmatics (Connection, Intent, Identity…) por ano
- [ ] Heatmap interativo de tópicos via Plotly
- [ ] Comparação entre eras (`pre-rafo`, `era rafo`, `late cosmere`)
- [ ] Exportar relatório estático em PDF

Para contribuir:

```bash
git checkout -b feat/minha-ideia
# … faça suas mudanças …
uv run marimo edit app.py     # teste local
uv run --extra build scripts/build.py  # teste o build
git commit -m "feat: minha ideia"
git push origin feat/minha-ideia
# Abra um Pull Request
```

---

## 📜 Licença

MIT — use, modifique, compartilhe.

## 🙏 Créditos

- **Brandon Sanderson** — por escrever livros que geram tantas perguntas boas
- **17th Shard** e a comunidade do **Coppermind** — pelo corpus de WOBs
- **Time do marimo** — pela ferramenta que torna tudo isso possível

---

<p align="center">
  <em>"The most important step a man can take. It's not the first one, is it?<br>
  It's the next one. Always the next step."</em><br>
  — Dalinar Kholin, <strong>Oathbringer</strong>
</p>
