FROM python:3.11-slim-bookworm

# Не даем Python писать .pyc файлы и буферизировать вывод (важно для логов)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_HTTP_TIMEOUT=1000
ENV UV_RETRIES=10

COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && uv pip install --system --no-cache -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8010

# Используем прямой запуск через python
CMD ["python", "main.py"]