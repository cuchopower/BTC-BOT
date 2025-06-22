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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import requests
from datetime import datetime

app = FastAPI()

TELEGRAM_TOKEN = '7666801859:AAFPwyWI_gPtqJO9CxJzUHyi1hu9eEQAj-c'
CHAT_ID = '7361418502'
historial = []

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
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=200)
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
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['vol_anormal'] = df['volume'] > (df['volume'].rolling(20).mean() * 1.2)

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

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        last_row = X_scaled[-1].reshape(1, -1)
        pred = model.predict(last_row)[0]
        probas = model.predict_proba(last_row)[0]
        max_proba = max(probas)
        pred = 1 if max_proba < 0.5 else pred
        pred_label = {0: "🔻 VENTA", 1: "⏸️ NEUTRO", 2: "🔺 COMPRA"}[pred]

        price = df['close'].iloc[-1]

        response = {
            "signal": pred_label,
            "confidence": f"{max_proba:.2%}",
            "price": price,
            "accuracy": round(acc * 100, 2),
            "f1_score": round(f1 * 100, 2)
        }

        if pred != 1:  # Solo calcular TP y SL si es COMPRA o VENTA
            tp = round(price + 2 * df['atr'].iloc[-1], 2)
            sl = round(price - 1.5 * df['atr'].iloc[-1], 2)
            response["tp"] = tp
            response["sl"] = sl

            mensaje = f"Señal: {pred_label}\nConfianza: {max_proba:.2%}\nPrecio: ${price:.2f}\nTP: ${tp}\nSL: ${sl}"
            enviar_telegram(mensaje)

        historial.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "signal": pred_label,
            "price": round(price, 2),
            "confidence": f"{max_proba:.2%}"
        })
        if len(historial) > 5:
            historial.pop(0)

        return response

    except Exception as e:
        print("❌ Error en /signal:", str(e))
        return {"signal": "⏸️ NEUTRO", "confidence": "0%", "price": 0}

@app.get("/history")
async def get_history():
    return historial

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()














  



