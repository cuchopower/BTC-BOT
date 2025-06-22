from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import requests

app = FastAPI()

TELEGRAM_TOKEN = '7666801859:AAFPwyWI_gPtqJO9CxJzUHyi1hu9eEQAj-c'
CHAT_ID = '7361418502'

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("❌ Error al enviar Telegram:", str(e))

@app.get("/signal")
async def get_signal():
    try:
        exchange = ccxt.kucoin()
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=300)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Indicadores técnicos
        df["rsi"] = RSIIndicator(close=df["close"]).rsi()
        df["ema20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
        df["ema50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        macd = MACD(close=df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["atr"] = atr.average_true_range()
        df["trend"] = np.where(df["ema20"] > df["ema50"], 1, np.where(df["ema20"] < df["ema50"], -1, 0))

        # Indicador líder: On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # Velas japonesas básicas: engulfing
        df['bullish_engulfing'] = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
        df['bearish_engulfing'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))

        # Patrón chartista simple: Doble fondo (2 mínimos cercanos y rebote)
        df['double_bottom'] = (df['low'].shift(2) > df['low'].shift(1)) & (df['low'].shift(1) < df['low']) & (df['close'] > df['close'].shift(1))

        # Volumen anormal: cuando el volumen es 1.5x mayor que el promedio
        df['vol_anormal'] = df['volume'] > (df['volume'].rolling(20).mean() * 1.5)

        # Señales
        df["signal"] = 0
        df.loc[(df["macd"] > df["macd_signal"]) & (df["rsi"] < 50) & (df["trend"] == 1) & df['vol_anormal'], "signal"] = 1
        df.loc[(df["macd"] < df["macd_signal"]) & (df["rsi"] > 50) & (df["trend"] == -1) & df['vol_anormal'], "signal"] = -1

        df.dropna(inplace=True)
        if len(df) < 100:
            return {"signal": "⏸️ NEUTRO", "confidence": "0%", "price": df['close'].iloc[-1]}

        X = df[["rsi", "ema20", "ema50", "macd", "macd_signal", "atr", "trend", "obv"]]
        y = df["signal"].replace({-1: 0, 0: 1, 1: 2})

        if y.nunique() < 2:
            return {"signal": "⏸️ NEUTRO", "confidence": "0%", "price": df['close'].iloc[-1]}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
        model.fit(X_scaled, y)

        last_row = X_scaled[-1].reshape(1, -1)
        pred = model.predict(last_row)[0]
        probas = model.predict_proba(last_row)[0]
        max_proba = max(probas)
        pred = 1 if max_proba < 0.6 else pred
        pred_label = {0: "🔻 VENTA", 1: "⏸️ NEUTRO", 2: "🔺 COMPRA"}[pred]

        mensaje = f"Señal: {pred_label}\nConfianza: {max_proba:.2%}\nPrecio: {df['close'].iloc[-1]:.2f}"
        if pred != 1:
            enviar_telegram(mensaje)

        return {
            "signal": pred_label,
            "confidence": f"{max_proba:.2%}",
            "price": df["close"].iloc[-1]
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()













  



