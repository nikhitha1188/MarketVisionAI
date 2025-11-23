# app.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Any

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Reduce TF logging
tf.get_logger().setLevel("ERROR")

# ---------------------------
# Utility: Technical indicators
# ---------------------------
def sma(series: pd.Series, window: int):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_rsi(series: pd.Series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_volatility(series: pd.Series, window=20):
    return series.pct_change().rolling(window=window, min_periods=1).std()

def compute_return_on_investment(series: pd.Series, period=10):
    # simple % change over period
    return series.pct_change(periods=period)

# ---------------------------
# StockEvaluator (Fundamentals)
# ---------------------------
class StockEvaluator:
    def __init__(self):
        pass

    def evaluate(self, info: dict) -> Dict[str, Any]:
        # prepare defaults to avoid missing keys
        pe = info.get("trailingPE") or info.get("forwardPE") or None
        eps = info.get("trailingEps") or info.get("forwardEps") or 0.0
        div_yield = info.get("dividendYield")
        market_cap = info.get("marketCap") or 0
        debt_to_equity = info.get("debtToEquity") or info.get("debtToEquity") or 1.0
        roe = info.get("returnOnEquity") or 0.0
        roa = info.get("returnOnAssets") or 0.0
        current_ratio = info.get("currentRatio") or 1.0
        quick_ratio = info.get("quickRatio") or 1.0
        book_value = info.get("bookValue") or 0.0

        # convert dividendYield from fraction to percentage if present
        if div_yield is not None:
            div_yield_pct = div_yield * 100
        else:
            div_yield_pct = 0.0

        # Evaluate textual feedback
        feedback = {}
        # P/E
        if pe is None:
            feedback['priceEarningsRatio'] = "P/E not available"
            pe_val = 999
        else:
            pe_val = float(pe)
            if pe_val < 15:
                feedback['priceEarningsRatio'] = f"P/E {pe_val:.2f}: potentially undervalued"
            elif pe_val <= 30:
                feedback['priceEarningsRatio'] = f"P/E {pe_val:.2f}: in normal range"
            else:
                feedback['priceEarningsRatio'] = f"P/E {pe_val:.2f}: relatively expensive"

        # EPS
        eps_val = float(eps) if eps is not None else 0.0
        if eps_val < 1:
            feedback['earningsPerShare'] = f"EPS {eps_val:.2f}: low"
        elif eps_val < 5:
            feedback['earningsPerShare'] = f"EPS {eps_val:.2f}: moderate"
        else:
            feedback['earningsPerShare'] = f"EPS {eps_val:.2f}: strong"

        # Dividend yield
        if div_yield_pct < 1:
            feedback['dividendYield'] = f"{div_yield_pct:.2f}%: low or none"
        elif div_yield_pct <= 3:
            feedback['dividendYield'] = f"{div_yield_pct:.2f}%: moderate"
        else:
            feedback['dividendYield'] = f"{div_yield_pct:.2f}%: attractive"

        # Market Cap
        cap = float(market_cap)
        if cap >= 1e12:
            feedback['marketCap'] = f"${cap/1e12:.2f}T: mega-cap"
        elif cap >= 1e9:
            feedback['marketCap'] = f"${cap/1e9:.2f}B: large-cap"
        else:
            feedback['marketCap'] = f"${cap/1e6:.2f}M: small/medium cap"

        # Debt to equity
        de_val = float(debt_to_equity)
        if de_val < 0.5:
            feedback['debtToEquityRatio'] = f"{de_val:.2f}: low leverage"
        elif de_val <= 1.5:
            feedback['debtToEquityRatio'] = f"{de_val:.2f}: moderate leverage"
        else:
            feedback['debtToEquityRatio'] = f"{de_val:.2f}: high leverage (risk)"

        # ROE & ROA
        roe_val = float(roe) if roe is not None else 0.0
        roa_val = float(roa) if roa is not None else 0.0
        feedback['returnOnEquity'] = f"{roe_val*100:.1f}%"
        feedback['returnOnAssets'] = f"{roa_val*100:.1f}%"

        # Liquidity
        feedback['currentRatio'] = f"{current_ratio:.2f}"
        feedback['quickRatio'] = f"{quick_ratio:.2f}"

        feedback['bookValuePerShare'] = f"{book_value:.2f}"

        # overall rating (0-100) - simplified weighted scoring
        score = 0
        # P/E weight (15)
        if pe is not None:
            if pe_val < 15: score += 15
            elif pe_val <= 25: score += 10
            elif pe_val <= 35: score += 5
        # EPS (15)
        if eps_val >= 5: score += 15
        elif eps_val >= 2: score += 10
        elif eps_val >= 1: score += 5
        # ROE (15)
        if roe_val >= 0.15: score += 15
        elif roe_val >= 0.10: score += 10
        elif roe_val >= 0.08: score += 5
        # ROA (10)
        if roa_val >= 0.10: score += 10
        elif roa_val >= 0.05: score += 6
        elif roa_val >= 0.03: score += 3
        # Liquidity (10)
        if current_ratio >= 1.5 and quick_ratio >= 1.2: score += 10
        elif current_ratio >= 1.0 and quick_ratio >= 0.8: score += 6
        # Debt (10)
        if de_val <= 0.5: score += 10
        elif de_val <= 1.0: score += 7
        elif de_val <= 1.5: score += 4
        # Dividend (5)
        if div_yield_pct >= 3: score += 5
        elif div_yield_pct >= 1.5: score += 3
        # Market cap stability (5)
        if cap >= 1e11: score += 5

        overall_rating = min(int(score), 100)

        return {
            "feedback": feedback,
            "overallRating": overall_rating,
            "raw": {
                "priceEarningsRatio": pe_val,
                "earningsPerShare": eps_val,
                "dividendYieldPct": div_yield_pct,
                "marketCap": cap,
                "debtToEquityRatio": de_val,
                "returnOnEquity": roe_val,
                "returnOnAssets": roa_val,
                "currentRatio": current_ratio,
                "quickRatio": quick_ratio,
                "bookValuePerShare": book_value
            }
        }

# ---------------------------
# Stock Price Predictor with technical features + LSTM
# ---------------------------
class StockPricePredictor:
    def __init__(self, sequence_length: int = 60, epochs: int = 8, batch_size: int = 32):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.df = None
        self.predictions = None

    def fetch_data(self, symbol: str, period: str = "3y"):
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No historical data found for {symbol}")
        df = df.dropna().copy()
        self.df = df
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["SMA_10"] = sma(df["Close"], 10)
        df["SMA_20"] = sma(df["Close"], 20)
        df["SMA_50"] = sma(df["Close"], 50)
        df["EMA_12"] = ema(df["Close"], 12)
        df["EMA_26"] = ema(df["Close"], 26)
        macd_line, macd_signal, macd_hist = compute_macd(df["Close"])
        df["MACD"] = macd_line
        df["MACD_Signal"] = macd_signal
        df["MACD_Hist"] = macd_hist
        df["RSI_14"] = compute_rsi(df["Close"], 14)
        df["Volatility_20"] = compute_volatility(df["Close"], 20)
        df["ROI_5"] = compute_return_on_investment(df["Close"], 5)
        # fill small NaNs by forward/backward fill
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        return df

    def prepare_features(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        df_features = df[["Open", "High", "Low", "Close", "Volume",
                          "SMA_10", "SMA_20", "SMA_50",
                          "EMA_12", "EMA_26",
                          "MACD", "MACD_Signal", "MACD_Hist",
                          "RSI_14", "Volatility_20", "ROI_5"]].copy()

        # Fill any remaining inf/nan
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_features.fillna(method="ffill", inplace=True)
        df_features.fillna(method="bfill", inplace=True)
        df_features.fillna(0.0, inplace=True)

        self.feature_columns = df_features.columns.tolist()

        # scale
        scaled = self.scaler.fit_transform(df_features.values)

        X, y = [], []
        for i in range(self.sequence_length, scaled.shape[0]):
            X.append(scaled[i-self.sequence_length:i, :])
            # predict close price (we'll predict scaled close value index 3)
            y.append(scaled[i, 3])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.25))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.15))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model

    def train(self, symbol: str):
        df = self.fetch_data(symbol, period="3y")
        df = self.add_technical_indicators(df)
        X, y = self.prepare_features(df)

        if X.shape[0] < 50:
            # not enough samples to train
            raise ValueError("Not enough historical data to train model reliably")

        # split train/val
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        history = self.model.fit(X_train, y_train,
                                 validation_data=(X_val, y_val),
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 verbose=0)

        # Evaluate
        y_pred_scaled = self.model.predict(X_val, verbose=0).flatten()
        # to compute real R2/RMSE, inverse-scale only the close column
        # Prepare array to inverse transform: set predicted close in column index 3
        inv_pred = np.zeros((len(y_pred_scaled), len(self.feature_columns)))
        inv_true = np.zeros_like(inv_pred)
        inv_pred[:, 3] = y_pred_scaled
        inv_true[:, 3] = y_val
        inv_pred = self.scaler.inverse_transform(inv_pred)[:, 3]
        inv_true = self.scaler.inverse_transform(inv_true)[:, 3]

        r2 = r2_score(inv_true, inv_pred)
        rmse = np.sqrt(mean_squared_error(inv_true, inv_pred))

        return {
            "history": history.history,
            "validation_r2": float(r2),
            "validation_rmse": float(rmse),
            "last_close": float(df["Close"].iloc[-1]),
            "df": df  # returning df for further predictions (caller can drop it)
        }

    def predict(self, days: int = 10):
        if self.model is None or self.df is None:
            raise RuntimeError("Model not trained or data not loaded")

        df = self.df.copy()
        df = self.add_technical_indicators(df)
        # get the last sequence_length rows for features
        features_df = df[self.feature_columns].tail(self.sequence_length)
        scaled_seq = self.scaler.transform(features_df.values)
        seq = scaled_seq.copy()
        preds = []

        for _ in range(days):
            X_pred = seq.reshape(1, seq.shape[0], seq.shape[1])
            scaled_close_pred = float(self.model.predict(X_pred, verbose=0)[0, 0])

            # create next row by taking last row and replacing close with predicted scaled close
            next_row = seq[-1].copy()
            next_row[3] = scaled_close_pred  # column index 3 is Close
            # append to seq and drop first
            seq = np.vstack([seq[1:], next_row])

            # inverse transform only close value
            inv = np.zeros((1, len(self.feature_columns)))
            inv[0, 3] = scaled_close_pred
            pred_price = float(self.scaler.inverse_transform(inv)[0, 3])
            preds.append(pred_price)

        self.predictions = preds
        return preds

# ---------------------------
# Recommendation engine (combine fundamentals & technicals)
# ---------------------------
def generate_recommendation(fundamental_result: dict, technical_preds: List[float], df_with_indicators: pd.DataFrame) -> Dict[str, Any]:
    """
    Basic logic:
    - Trend measured by percentage change between first and last predicted price
    - Use moving averages on last known OHLC to detect short-term momentum
    - Use RSI to avoid buying into overbought (>70) or selling into oversold (<30)
    - Combine with fundamentals overallRating (0-100)
    """
    rec = {"recommendation": "HOLD", "confidence": "Low", "reasons": []}

    # fundamental strength
    rating = fundamental_result.get("overallRating", 50)
    if rating >= 75:
        fund_strength = "Strong"
    elif rating >= 50:
        fund_strength = "Moderate"
    else:
        fund_strength = "Weak"

    # predicted trend
    if not technical_preds or len(technical_preds) < 2:
        trend_pct = 0.0
    else:
        trend_pct = (technical_preds[-1] - technical_preds[0]) / technical_preds[0] * 100

    # last indicators from df
    last = df_with_indicators.iloc[-1]
    sma20 = last.get("SMA_20", np.nan)
    sma50 = last.get("SMA_50", np.nan)
    rsi = last.get("RSI_14", 50)
    macd = last.get("MACD", 0)
    macd_sig = last.get("MACD_Signal", 0)

    # Simple MA crossover signal
    ma_signal = None
    if np.isfinite(sma20) and np.isfinite(sma50):
        if sma20 > sma50:
            ma_signal = "bullish"
        elif sma20 < sma50:
            ma_signal = "bearish"

    # MACD signal
    macd_signal = None
    if macd > macd_sig:
        macd_signal = "bullish"
    elif macd < macd_sig:
        macd_signal = "bearish"

    # Compose signals
    reasons = []
    score = 0

    # fundamental adds to score
    if fund_strength == "Strong":
        score += 2
        reasons.append("Strong fundamentals")
    elif fund_strength == "Moderate":
        score += 1
        reasons.append("Moderate fundamentals")
    else:
        score -= 1
        reasons.append("Weak fundamentals")

    # trend
    if trend_pct > 4:
        score += 2
        reasons.append(f"Predicted upward trend ({trend_pct:.2f}%)")
    elif trend_pct > 1:
        score += 1
        reasons.append(f"Mild upward trend ({trend_pct:.2f}%)")
    elif trend_pct < -4:
        score -= 2
        reasons.append(f"Predicted downward trend ({trend_pct:.2f}%)")
    elif trend_pct < -1:
        score -= 1
        reasons.append(f"Mild downward trend ({trend_pct:.2f}%)")

    # MA signal
    if ma_signal == "bullish":
        score += 1
        reasons.append("20-day SMA above 50-day SMA (bullish)")
    elif ma_signal == "bearish":
        score -= 1
        reasons.append("20-day SMA below 50-day SMA (bearish)")

    # MACD
    if macd_signal == "bullish":
        score += 1
        reasons.append("MACD line above signal (bullish)")
    elif macd_signal == "bearish":
        score -= 1
        reasons.append("MACD line below signal (bearish)")

    # RSI caution
    if rsi > 70:
        score -= 1
        reasons.append(f"RSI {rsi:.1f} (overbought)")
    elif rsi < 30:
        score += 1
        reasons.append(f"RSI {rsi:.1f} (oversold)")

    # Final decision
    if score >= 4:
        rec["recommendation"] = "BUY"
        rec["confidence"] = "High"
    elif score >= 2:
        rec["recommendation"] = "BUY"
        rec["confidence"] = "Medium"
    elif score <= -4:
        rec["recommendation"] = "SELL"
        rec["confidence"] = "High"
    elif score <= -2:
        rec["recommendation"] = "SELL"
        rec["confidence"] = "Medium"
    else:
        rec["recommendation"] = "HOLD"
        rec["confidence"] = "Medium" if score != 0 else "Low"

    rec["reasons"] = reasons
    rec["score"] = score
    rec["trend_pct"] = round(trend_pct, 4)
    rec["fundamental_rating"] = rating
    return rec

# ---------------------------
# NextGenMarketAnalyzer - combine everything
# ---------------------------
class NextGenMarketAnalyzer:
    def __init__(self, lstm_sequence=60, lstm_epochs=8):
        self.evaluator = StockEvaluator()
        self.predictor = StockPricePredictor(sequence_length=lstm_sequence, epochs=lstm_epochs)

    def analyze(self, symbol: str, predict_days: int = 10) -> Dict[str, Any]:
        # 1) fetch fundamentals from yfinance
        ticker = yf.Ticker(symbol)
        info = {}
        try:
            info = ticker.info
        except Exception:
            info = {}

        fundamentals = self.evaluator.evaluate(info)

        # 2) Train predictor and evaluate
        train_result = None
        preds = []
        try:
            train_result = self.predictor.train(symbol)
            # predictor.train returns validation metrics and df; we kept df in train return
            df_for_preds = train_result.get("df")
            # ensure predictor's df is set
            self.predictor.df = df_for_preds
            preds = self.predictor.predict(days=predict_days)
        except Exception as e:
            # fallback: try to fetch last close and generate flat predictions
            last_close = None
            try:
                hist = ticker.history(period="1mo")
                last_close = float(hist["Close"].iloc[-1])
            except Exception:
                last_close = None
            if last_close is not None:
                preds = [last_close] * predict_days
            train_result = {"error": str(e)}

            # Build a df_for_preds minimal
            try:
                df_for_preds = self.predictor.add_technical_indicators(self.predictor.fetch_data(symbol, period="1y"))
            except Exception:
                df_for_preds = pd.DataFrame()

        # 3) Generate recommendation
        recommendation = generate_recommendation(fundamentals, preds, df_for_preds if not df_for_preds.empty else pd.DataFrame())

        # 4) pack detailed output
        output = {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat(),
            "fundamentals": fundamentals,
            "technical_backtest": {
                "validation_r2": train_result.get("validation_r2") if isinstance(train_result, dict) else None,
                "validation_rmse": train_result.get("validation_rmse") if isinstance(train_result, dict) else None
            },
            "predictions": [round(float(p), 4) for p in preds],
            "prediction_days": predict_days,
            "recommendation": recommendation
        }
        return output

# ---------------------------
# FASTAPI App
# ---------------------------
app = FastAPI(title="NextGen Market Analyzer (Fundamentals + Tech + LSTM)")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # frontend URLs
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, etc.
    allow_headers=["*"],          # allow all headers
)

analyzer = NextGenMarketAnalyzer(lstm_sequence=60, lstm_epochs=8)

@app.get("/analyze")
def analyze(symbol: str = Query(..., description="Stock ticker, e.g. AAPL"),
            days: int = Query(10, description="Days to predict (default 10)")):
    """
    Full analysis endpoint. Returns:
    - fundamentals and overall rating
    - technical validation stats (R2, RMSE)
    - predictions (next N days)
    - final recommendation (BUY/HOLD/SELL), confidence, reasons
    """
    try:
        result = analyzer.analyze(symbol, predict_days=days)
        return result
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

# Quick root endpoint
@app.get("/")
def root():
    return {"message": "NextGen Market Analyzer API. Use /analyze?symbol=XXX&days=Y"}

# ---------------------------
# Run with:
# python -m uvicorn app:app --reload
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
