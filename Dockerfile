# Базовий образ з Python
FROM python:3.13-slim

# Metadata
LABEL maintainer="dmytro.kosiachenko1@gmail.com"
LABEL description="RAG System for document Q&A"

# Встановлюємо системні залежності
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо файли залежностей
COPY requirements.txt .

# Встановлюємо Python залежності
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копіюємо весь проєкт
COPY . .

# Створюємо директорії для даних
RUN mkdir -p /app/data/documents \
             /app/data/indexes \
             /app/data/logs

# Встановлюємо змінні середовища
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose порт для API (якщо буде)
EXPOSE 8000

# Команда запуску за замовчуванням
CMD ["python", "main.py"]
