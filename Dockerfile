FROM python:3.12-slim

# uv para gerenciamento de pacotes
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copiar dependências primeiro (cache de layer)
COPY pyproject.toml uv.lock* ./

# Instalar dependências (incluindo as de build)
RUN uv sync --extra build --no-dev

# Copiar o resto do projeto
COPY . .

# Pré-computar e exportar
RUN uv run --extra build scripts/build.py

# Servir o site estático
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "-d", "_site"]