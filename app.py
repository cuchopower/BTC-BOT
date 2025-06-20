from fastapi import FastAPI
import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

def obtener_datos():
    exchange = ccxt.kucoin()
    df = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=500)
    df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calcular_indicadores(df):
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['ema20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    df['trend'] = np.where(df['ema20'] > df['ema50'], 1, np.where(df['ema20'] < df['ema50'], -1, 0))
    return df.dropna()

def entrenar_modelo(df):
    df['signal'] = 0
    df.loc[(df['macd'] > df['macd_signal']) & (df['rsi'] < 50) & (df['adx'] > 20) & (df['trend'] == 1), 'signal'] = 1
    df.loc[(df['macd'] < df['macd_signal']) & (df['rsi'] > 50) & (df['adx'] > 20) & (df['trend'] == -1), 'signal'] = -1
    df['target'] = df['signal'].replace({-1: 0, 0: 1, 1: 2})
    
    if df['target'].nunique() < 2:
        return None, None, None

    X = df[['rsi', 'ema20', 'ema50', 'macd', 'macd_signal', 'atr', 'trend', 'adx']]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    model.fit(X_scaled, y)
    return model, scaler, df

@app.get("/")
def home():
    return {"message": "Bot de Trading activo"}

@app.get("/signal")
def signal():
    try:
        df = obtener_datos()
        df = calcular_indicadores(df)
        model, scaler, df = entrenar_modelo(df)

        if model is None:
            return {"signal": "NEUTRO", "confidence": 0, "message": "No hay suficiente variedad para entrenar."}

        last_row = scaler.transform([df[['rsi', 'ema20', 'ema50', 'macd', 'macd_signal', 'atr', 'trend', 'adx']].iloc[-1]])
        pred = model.predict(last_row)[0]
        probas = model.predict_proba(last_row)[0]
        max_proba = max(probas)
        pred = 1 if max_proba < 0.6 else pred
        pred_label = {0: "🔻 VENTA", 1: "⏸️ NEUTRO", 2: "🔺 COMPRA"}[pred]

        return {
            "signal": pred_label,
            "confidence": f"{max_proba:.2%}",
            "price": round(df['close'].iloc[-1], 2)
        }
    except Exception as e:
        return {"error": str(e)}

  



