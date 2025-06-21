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

app = FastAPI()

@app.get("/signal")
async def get_signal():
    try:
        # ▶️ Datos desde KuCoin
        exchange = ccxt.kucoin()
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=300)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

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

        return {
            "signal": pred_label,
            "confidence": f"{max_proba:.2%}",
            "price": df["close"].iloc[-1]
        }

    except Exception as e:
        return {"error": str(e)}

# ▶️ Ruta que sirve el HTML (frontend)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()


  



