const axios = require('axios');
const bodyParser = require('body-parser');
const express = require('express');
const app = express();
const PORT = 3000;

app.use(bodyParser.json());

// Тестовый маршрут
app.get('/', (req, res) => {
    res.send('Node.js сервер работает!');
});

// Маршрут для обучения модели
app.post('/train', async (req, res) => {
    try {
        const response = await axios.post('http://ml-service:5000/train', req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Ошибка при обучении модели', details: error.message });
    }
});

// Маршрут для предсказаний
app.post('/predict', async (req, res) => {
    try {
        // Запрос данных с API ByBit
        const bybitResponse = await axios.get('https://api.bybit.com/v5/market/kline', {
            params: {
                category: 'linear',
                symbol: 'XRPUSDT',
                interval: '1',
                limit: 1000
            }
        });

        // Преобразование данных ByBit в формат, ожидаемый моделью
        const klineData = bybitResponse.data.result.list.map(candle => ({
            open: parseFloat(candle[1]),
            high: parseFloat(candle[2]),
            low: parseFloat(candle[3]),
            close: parseFloat(candle[4]),
            volume: parseFloat(candle[5])
        }));

        console.log('Prepared klineData:', klineData);  // Добавьте это перед отправкой данных

        // Отправка данных модели для предсказания
        const mlResponse = await axios.post('http://ml-service:5000/predict', {
            data: klineData
        });

        console.log('mlResponse data:', mlResponse.data);


        // Возврат предсказания клиенту
        res.json(mlResponse.data);
    } catch (error) {
        res.status(500).json({ error: 'Ошибка при получении предсказания', details: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Сервер запущен на http://localhost:${PORT}`);
});
