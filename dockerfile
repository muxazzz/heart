# Базовый образ Python
FROM python:3.11-slim

# Установка зависимостей системы (опционально, если нужно)
RUN apt-get update && apt-get install -y build-essential

# Установка рабочей директории
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение
COPY . .

# Открываем порт
EXPOSE 8000

# Запускаем приложение через Uvicorn
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
