# Используем базовый образ Node.js
FROM node:18

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY package*.json ./
COPY app.js ./

# Устанавливаем зависимости
RUN npm install

# Указываем порт
EXPOSE 3000

# Запускаем сервер
CMD ["npm", "start"]
