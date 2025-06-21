from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import ccxt
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# ⚙️ Parámetros
capital = 10000
risk_per_trade = 0.02
atr_multiplier = 2

# 🔔 Configuración de Telegram
TELEGRAM_TOKEN = '7666801859:AAFPwyWI_gPtqJO9CxJzUHyi1hu9eEQAj-c'
CHAT_ID = '7361418502'  # Reemplaza por el tuyo

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("❌ Error al enviar Telegram:", str(e))

def obtener_datos_kucoin(activo='BTC/USDT', timeframe='5m', limit=500):
    exchange = ccxt.kucoin()
    ohlcv = exchange.fetch_ohlcv(activo, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def analizar_senal(df):
    if len(df) < 100:
        return None

    # Indicadores
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['ema20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    df['trend'] = np.where(df['ema20'] > df['ema50'], 1, np.where(df['ema20'] < df['ema50'], -1, 0))
    df.dropna(inplace=True)

    if len(df) < 100:
        return None

    # Señales
    df['signal'] = 0
    df.loc[(df['macd'] > df['macd_signal']) & (df['rsi'] < 50) & (df['adx'] > 20) & (df['trend'] == 1), 'signal'] = 1
    df.loc[(df['macd'] < df['macd_signal']) & (df['rsi'] > 50) & (df['adx'] > 20) & (df['trend'] == -1), 'signal'] = -1

    # ML
    X = df[['rsi', 'ema20', 'ema50', 'macd', 'macd_signal', 'atr', 'trend', 'adx']]
    y = df['signal'].replace({-1: 0, 0: 1, 1: 2})

    if y.nunique() < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    model.fit(X_scaled, y)

    last_row = X_scaled[-1].reshape(1, -1)
    pred = model.predict(last_row)[0]
    probas = model.predict_proba(last_row)[0]
    max_proba = max(probas)
    pred = 1 if max_proba < 0.6 else pred
    pred_label = {0: "🔻 VENTA", 1: "⏸️ NEUTRO", 2: "🔺 COMPRA"}[pred]

    precio_actual = df['close'].iloc[-1]

    if pred in [0, 2]:
        atr_valor = df['atr'].iloc[-1]
        tp = precio_actual + atr_valor * atr_multiplier if pred == 2 else precio_actual - atr_valor * atr_multiplier
        sl = precio_actual - atr_valor * atr_multiplier if pred == 2 else precio_actual + atr_valor * atr_multiplier
        risk_distance = abs(precio_actual - sl)
        position_size = (capital * risk_per_trade) / risk_distance
    else:
        tp = sl = position_size = None

    return {
        "signal": pred_label,
        "confidence": f"{max_proba:.2%}",
        "price": round(precio_actual, 2),
        "take_profit": round(tp, 2) if tp else None,
        "stop_loss": round(sl, 2) if sl else None,
        "position_size": round(position_size, 2) if position_size else None
    }

@app.get("/")
def root():
    return HTMLResponse("""
    <html>
    <head><title>Bot de Trading</title></head>
    <body style="font-family:Arial;text-align:center;">
        <h2>🤖 Bot de Trading en Render</h2>
        <p>Ir a <a href="/signal">/signal</a> para ver la última señal de BTC/USDT (5m)</p>
    </body>
    </html>
    """)

@app.get("/signal")
def signal():
    df = obtener_datos_kucoin()
    resultado = analizar_senal(df)

    if resultado:
        if resultado["signal"] != "⏸️ NEUTRO":
            mensaje = f"🚨 Señal detectada: {resultado['signal']}\nPrecio: {resultado['price']}\nConfianza: {resultado['confidence']}"
            enviar_telegram(mensaje)
        return resultado
    else:
        return {"signal": "⏸️ NEUTRO", "confidence": "N/A"}


  



