Example check https://api.bybit.com/v5/market/kline?category=linear&symbol=XRPUSDT&interval=1&limit=1024&start=1733011200000&end=1737368094936

Howto use this code:

docker compose -f ./docker-compose.yml up --build

wait

call 2 post requests:

1. http://localhost:3000/train
2. http://localhost:3000/predict
