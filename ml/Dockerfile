# Используем базовый образ Python
FROM python:3.9

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY model.py ./
COPY requirements.txt ./

# Устанавливаем зависимости
RUN pip install -r requirements.txt

# Указываем порт
EXPOSE 5000

# Запускаем Flask-сервис
CMD ["python", "model.py"]
