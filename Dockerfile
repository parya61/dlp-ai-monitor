# ШАГ 1: Базовый образ
# Берём официальный Python 3.10 (slim = урезанная версия, меньше размер)
# ЗАЧЕМ: У тебя Python 3.13, но для production лучше 3.10 (стабильнее)
FROM python:3.10-slim

# ШАГ 2: Метаданные (опционально, но хорошая практика)
LABEL maintainer="your-email@example.com"
LABEL description="DLP AI Monitor - ML система для обнаружения утечек данных"

# ШАГ 3: Рабочая директория внутри контейнера
# Все команды будут выполняться отсюда
WORKDIR /app

# ШАГ 4: Устанавливаем системные зависимости (если нужны)
# ЗАЧЕМ: Некоторые Python пакеты требуют компиляторы
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ШАГ 5: Копируем requirements.txt ПЕРВЫМ
# ЗАЧЕМ: Docker кэширует слои. Если код изменился, а requirements.txt нет - 
# не будет переустанавливать все пакеты (экономия времени!)
COPY requirements.txt .

# ШАГ 6: Устанавливаем Python зависимости
# --no-cache-dir = не сохранять кэш pip (меньше размер образа)
# --upgrade pip = обновляем pip до последней версии
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ШАГ 7: Копируем весь код проекта
# . . означает "скопировать всё из текущей папки в /app"
# .dockerignore исключит ненужные файлы
COPY . .

# ШАГ 8: Создаём директории для данных и моделей
# ЗАЧЕМ: Контейнер изолирован, нужно явно создать папки
RUN mkdir -p data/raw data/processed models logs

# ШАГ 9: Открываем порт 8000
# ЗАЧЕМ: FastAPI будет слушать на этом порту
# ВАЖНО: Это только документация! Реальный порт настраивается в docker-compose
ENV PYTHONPATH=/app

# ШАГ 9: Открываем порт 8000
EXPOSE 8000

# ШАГ 10: Команда запуска при старте контейнера
# uvicorn запускает FastAPI сервер
# --host 0.0.0.0 = слушать на всех интерфейсах (чтобы достучаться снаружи)

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]