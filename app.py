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

# ⚙️ Parámetros de Telegram
TELEGRAM_TOKEN = '7666801859:AAFPwyWI_gPtqJO9CxJzUHyi1hu9eEQAj-c'
CHAT_ID = '7361418502'

# ✅ Obtener datos desde BingX o KuCoin

def get_ohlcv(symbol="BTC/USDT", timeframe="5m", limit=300):
    if symbol in ["NAS100USDT", "US30USDT", "SPX500USDT", "XAUUSDT", "EURUSD", "GBPUSD"]:
        url = "https://open-api.bingx.com/openApi/market/kline"
        params = {"symbol": symbol, "interval": timeframe, "limit": limit}
        response = requests.get(url, params=params)
        data = response.json()
        if data["code"] != 0:
            raise Exception("BingX error: " + data["msg"])
        df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
    else:
        exchange = ccxt.kucoin()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
    return df

# 📡 Enviar notificación

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
        symbol = "NAS100USDT"  # Cambiar aquí para probar otro activo
        df = get_ohlcv(symbol, timeframe="5m", limit=300)

        # ▶️ Indicadores
        df["rsi"] = RSIIndicator(close=df["close"]).rsi()
        df["ema20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
        df["ema50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        macd = MACD(close=df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["atr"] = atr.average_true_range()
        df["trend"] = np.where(df["ema20"] > df["ema50"], 1, np.where(df["ema20"] < df["ema50"], -1, 0))

        df["signal"] = 0
        df.loc[(df["macd"] > df["macd_signal"]) & (df["rsi"] < 50) & (df["trend"] == 1), "signal"] = 1
        df.loc[(df["macd"] < df["macd_signal"]) & (df["rsi"] > 50) & (df["trend"] == -1), "signal"] = -1

        df.dropna(inplace=True)
        if len(df) < 100:
            return {"signal": "⏸️ NEUTRO", "confidence": "0%", "price": df['close'].iloc[-1]}

        X = df[["rsi", "ema20", "ema50", "macd", "macd_signal", "atr", "trend"]]
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

        mensaje = f"{symbol}\n📈 Señal: {pred_label}\n🤖 Confianza: {max_proba:.2%}\n💵 Precio: {df['close'].iloc[-1]:.2f}"
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



  



