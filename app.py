# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import uvicorn

app = FastAPI()

class SignalRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"

@app.get("/")
def root():
    return {"message": "Bot de Trading activo"}

@app.post("/signal")
def signal(req: SignalRequest):
    try:
        exchange = ccxt.kucoin()
        ohlcv = exchange.fetch_ohlcv(req.symbol, timeframe=req.timeframe, limit=500)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        if df.empty or len(df) < 100:
            return {"error": "No hay suficientes datos"}

        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        df['ema20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        df['atr'] = atr.average_true_range()
        df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
        df['trend'] = np.where(df['ema20'] > df['ema50'], 1, np.where(df['ema20'] < df['ema50'], -1, 0))

        df['signal'] = 0
        df.loc[(df['macd'] > df['macd_signal']) & (df['rsi'] < 50) & (df['adx'] > 20) & (df['trend'] == 1), 'signal'] = 1
        df.loc[(df['macd'] < df['macd_signal']) & (df['rsi'] > 50) & (df['adx'] > 20) & (df['trend'] == -1), 'signal'] = -1

        df.dropna(inplace=True)
        if df.empty or df['signal'].nunique() < 2:
            return {"signal": "⏸️ Neutro (sin suficientes señales)"}

        X = df[['rsi', 'ema20', 'ema50', 'macd', 'macd_signal', 'atr', 'trend', 'adx']]
        y = df['signal'].replace({-1: 0, 0: 1, 1: 2})
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        model.fit(X_scaled, y)

        last_row = X_scaled[-1].reshape(1, -1)
        pred = model.predict(last_row)[0]
        probas = model.predict_proba(last_row)[0]
        max_proba = float(np.max(probas))
        pred = 1 if max_proba < 0.6 else pred

        label = {0: "🔻 VENTA", 1: "⏸️ NEUTRO", 2: "🔺 COMPRA"}[pred]
        return {"signal": label, "confidence": f"{max_proba:.2%}"}

    except Exception as e:
        return {"error": str(e)}
  



