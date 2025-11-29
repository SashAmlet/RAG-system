# Базовий образ з Python
FROM python:3.13-slim

# Встановлюємо системні залежності
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Робоча директорія
WORKDIR /app

# Копіюємо requirements
COPY requirements.txt .

# Встановлюємо Python залежності
RUN pip install --no-cache-dir -r requirements.txt

# Завантажуємо NLTK дані (якщо потрібні)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Копіюємо код проєкту
COPY . .

# Створюємо директорії для даних
RUN mkdir -p data/raw data/processed data/indexes

# Змінні середовища
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Порт для можливого веб-інтерфейсу (якщо додасте пізніше)
EXPOSE 8000

# Точка входу
CMD ["python", "main.py"]