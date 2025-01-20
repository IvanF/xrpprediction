import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import time

# Инициализация Flask приложения
app = Flask(__name__)

# Глобальные переменные
model = None
scaler = MinMaxScaler()

# Функция для получения данных с ByBit
def fetch_bybit_data():
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": "XRPUSDT",
        "interval": "1",
        "limit": "1000"
    }

    # Начальное время (текущее время в миллисекундах)
    end_time = int(time.time() * 1000)  # Текущее время в миллисекундах
    start_time = end_time - (60 * 1000 * 1000)  # Около 10 000 минут назад

    all_data = []  # Для хранения всех свечей

    while len(all_data) < 10000:
        # Установка временного диапазона
        params["start"] = start_time
        params["end"] = end_time

        # Выполнение запроса
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Ошибка получения данных от ByBit: {response.status_code}")

        data = response.json()
        candles = data.get("result", {}).get("list", [])

        if not candles:
            break  # Если данные закончились, выходим из цикла

        # Преобразуем в список словарей
        formatted_data = [
            {
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            }
            for candle in candles
        ]

        all_data.extend(formatted_data)

        # Обновляем `end_time` для следующей итерации
        end_time = int(candles[0][0])  # Время открытия самой старой свечи в миллисекундах

        # Если свечей меньше 1000, значит мы дошли до конца
        if len(candles) < 1000:
            break

    # Обрезаем данные до 10 000 свечей, если их больше
    all_data = all_data[:10000]

    if len(all_data) == 0:
        raise ValueError("Полученные данные пусты")

    return all_data

# Функция для вычисления уровней поддержки и сопротивления
def calculate_support_resistance(df, window=20):
    support = df['low'].rolling(window=window).min()
    resistance = df['high'].rolling(window=window).max()
    return support, resistance

# Функция для вычисления тренда (с использованием простых скользящих средних)
def calculate_trend(df):
    short_window = 20
    long_window = 50

    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()

    # Определение тренда на основе скользящих средних
    if df['short_ma'].iloc[-1] > df['long_ma'].iloc[-1]:
        trend = 'uptrend'
    elif df['short_ma'].iloc[-1] < df['long_ma'].iloc[-1]:
        trend = 'downtrend'
    else:
        trend = 'neutral'

    return trend

# Подготовка данных
def preprocess_data(data):
    if not isinstance(data, list):
        raise ValueError("Ожидается список данных")

    df = pd.DataFrame(data)
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Отсутствует столбец: {column}")

    df['mid_price'] = (df['high'] + df['low']) / 2
    X = df[['open', 'close', 'mid_price', 'volume']]
    y = df['close'].shift(-1).fillna(df['close'])

    # Масштабирование данных
    X = scaler.fit_transform(X)
    X = np.array(X)

    return X, y, df

# Обучение модели с распараллеливанием на всех доступных ядрах CPU
def train_model(data):
    global model
    X, y, df = preprocess_data(data)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # Получаем количество доступных ядер
    num_threads = os.cpu_count()
    print(f"Using {num_threads} CPU threads")

    # Устанавливаем параметры для параллельного обучения
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)  # Внутренние параллельные операции
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)  # Параллельные операции между слоями

    # Создание модели
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(5)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Обучение модели
    model.fit(X, y, epochs=10, batch_size=2, validation_split=0.2)

    # Сохранение модели
    model.save('model.h5')
    return {"message": "Model trained successfully"}

# Маршрут для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    global model
    if not model and os.path.exists('model.h5'):
        from tensorflow.keras.models import load_model
        model = load_model('model.h5')
    if not model:
        return jsonify({"error": "Model not trained yet"})

    try:
        # Извлекаем данные из запроса
        data = request.json['data']

        # Преобразуем данные в формат, который ожидает модель
        df = pd.DataFrame(data)
        required_columns = ['open', 'close', 'high', 'low', 'volume']

        # Проверяем наличие необходимых столбцов
        for column in required_columns:
            if column not in df.columns:
                return jsonify({"error": f"Missing column: {column}"})

        # Добавляем столбец mid_price
        df['mid_price'] = (df['high'] + df['low']) / 2
        X = df[['open', 'close', 'mid_price', 'volume']]

        # Масштабируем данные
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Получение предсказания
        prediction = model.predict(X_scaled)

        # Вычисление уровней поддержки и сопротивления
        support, resistance = calculate_support_resistance(df)

        # Определение тренда
        trend = calculate_trend(df)

        # Получение последней цены
        last_price = df['close'].iloc[-1]

        # Рассчитываем, насколько цена близка к уровню сопротивления или поддержки
        distance_to_support = (last_price - support.iloc[-1]) / support.iloc[-1] * 100
        distance_to_resistance = (resistance.iloc[-1] - last_price) / resistance.iloc[-1] * 100
        current_price = df['close'].iloc[-1]

        return jsonify({

            "prediction": float(prediction[0][0]),
            "trend": trend,
            "distance_to_support": distance_to_support,
            "distance_to_resistance": distance_to_resistance,
            "support_price": float(support.iloc[-1]),
            "resistance_price": float(resistance.iloc[-1]),
            "current_price": current_price
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Маршрут для обучения модели
@app.route('/train', methods=['POST'])
def train():
    try:
        # Получение реальных данных
        data = fetch_bybit_data()
        result = train_model(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
