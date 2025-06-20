from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime

app = FastAPI()

capital = 10000
risk_per_trade = 0.02
atr_multiplier = 2

def fetch_data(activo, timeframe):
    if '/' in activo:
        exchange = ccxt.kucoin()
        ohlcv = exchange.fetch_ohlcv(activo, timeframe=timeframe, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    else:
        return None

def compute_indicators(df):
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['ema20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['adx'] = adx.adx()
    df['trend'] = np.where(df['ema20'] > df['ema50'], 1, -1)
    return df.dropna()

def create_labels(df):
    df['future_close'] = df['close'].shift(-3)
    df['label'] = np.where(df['future_close'] > df['close'], 2, 0)
    return df.dropna()

def prepare_lstm_data(df, sequence_length=20):
    features = ['rsi', 'ema20', 'ema50', 'macd', 'macd_signal', 'atr', 'adx', 'trend']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[features].iloc[i:i+sequence_length].values)
        y.append(df['label'].iloc[i+sequence_length])
    return np.array(X), np.array(y), df

def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@app.get("/")
def root():
    return {"message": "Bot de Trading activo"}

@app.api_route("/signal", methods=["GET", "POST"])
async def get_signal(request: Request):
    if request.method == "POST":
        data = await request.json()
        activo = data.get("activo", "BTC/USDT")
        timeframe = data.get("timeframe", "5m")
    else:
        params = request.query_params
        activo = params.get("activo", "BTC/USDT")
        timeframe = params.get("timeframe", "5m")

    df = fetch_data(activo, timeframe)
    if df is None or df.empty:
        return JSONResponse(content={"error": "Datos no disponibles"}, status_code=400)

    df = compute_indicators(df)
    df = create_labels(df)

    if df.empty or len(df) < 40:
        return JSONResponse(content={"error": "No hay suficientes datos"}, status_code=400)

    X, y, df = prepare_lstm_data(df)
    model = build_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    last_input = X[-1].reshape(1, X.shape[1], X.shape[2])
    pred = model.predict(last_input)[0]
    pred_class = int(np.argmax(pred))
    confidence = float(pred[pred_class])
    label_map = {0: "🔻 VENTA", 1: "⏸️ NEUTRO", 2: "🔺 COMPRA"}

    price = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    if pred_class in [0, 2]:
        tp = price + atr * atr_multiplier if pred_class == 2 else price - atr * atr_multiplier
        sl = price - atr * atr_multiplier if pred_class == 2 else price + atr * atr_multiplier
        risk = abs(price - sl)
        size = (capital * risk_per_trade) / risk
    else:
        tp = sl = size = None

    return {
        "activo": activo,
        "timeframe": timeframe,
        "precio": round(price, 2),
        "señal": label_map[pred_class],
        "confianza": f"{confidence:.2%}",
        "take_profit": round(tp, 2) if tp else None,
        "stop_loss": round(sl, 2) if sl else None,
        "tamaño_sugerido": round(size, 2) if size else None,
        "fecha": datetime.utcnow().isoformat()
    }

  



