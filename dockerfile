# Используем официальный образ Python в качестве базового
FROM python:3.10-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем requirements.txt в рабочую директорию
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в рабочую директорию
COPY . .

# Указываем команду запуска Streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["web_script.py"]
