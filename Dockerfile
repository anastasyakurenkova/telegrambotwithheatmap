# Используем официальный образ Python 3.9
FROM python:3.9-slim

# Устанавливаем зависимости системы
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл requirements.txt в рабочую директорию
COPY requirements.txt .

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы вашего приложения в контейнер
COPY . .

# Указываем команду, которая будет запускаться при старте контейнера
CMD ["python", "telegram_bot_scripts.py"]
