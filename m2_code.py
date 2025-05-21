# milestone2_full_backtest.py

import os
import time
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ─── Configuration ─────────────────────────────────────────────────────────────
TICKERS         = ['SPY', 'QQQ', 'AAPL', 'TSLA']
INITIAL_CAP     = 100_000.00
ALLOC_PER_TRADE = 0.05
DELTA           = 0.5
WINDOW          = 20
MODEL_FILE      = 'xgb_odte_model.pkl'

# ─── Connect to IBKR ────────────────────────────────────────────────────────────
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)

# ─── Robust Historical Data Fetch ───────────────────────────────────────────────
def fetch_data(symbol, duration='14 D', maxRetries=3):
    contract = Stock(symbol, 'SMART', 'USD', primaryExchange='ARCA')
    ib.qualifyContracts(contract)

    # Retry logic
    for attempt in range(1, maxRetries + 1):
        bars = ib.reqHistoricalData(
            contract, endDateTime='', durationStr=duration,
            barSizeSetting='1 min', whatToShow='TRADES', useRTH=True
        )
        if bars:
            return util.df(bars).set_index('date')

        wait = 2 ** attempt
        print(f"⚠️ Timeout fetching {symbol} for {duration}, retrying in {wait}s…")
        time.sleep(wait)

    # Fallback: chunk into 30-day slices
    print(f"⚠️ Fallback: splitting {duration} into 30-day chunks for {symbol}")
    total_days = int(duration.split()[0])
    parts = []
    endDate = ''
    for start in range(0, total_days, 30):
        chunk = ib.reqHistoricalData(
            contract, endDateTime=endDate, durationStr='30 D',
            barSizeSetting='1 min', whatToShow='TRADES', useRTH=True
        )
        if chunk:
            df_chunk = util.df(chunk).set_index('date')
            parts.append(df_chunk)
            endDate = df_chunk.index[0].strftime('%Y%m%d %H:%M:%S')
        else:
            print(f"⚠️ Failed to fetch chunk {start}-{start+30} D for {symbol}")
    if parts:
        return pd.concat(parts).sort_index()

    raise RuntimeError(f"Unable to fetch historical data for {symbol}")

# ─── Indicator Computation ─────────────────────────────────────────────────────
def compute_indicators(df):
    df['vwap']    = (df['close'] * df['volume']).rolling(WINDOW).sum() / df['volume'].rolling(WINDOW).sum()

    delta_px      = df['close'].diff()
    gain          = delta_px.clip(lower=0)
    loss          = -delta_px.clip(upper=0)
    avg_gain      = gain.rolling(14).mean()
    avg_loss      = loss.rolling(14).mean()
    rs            = avg_gain / (avg_loss + 1e-5)
    df['rsi']     = 100 - (100 / (1 + rs))

    exp12         = df['close'].ewm(span=12, adjust=False).mean()
    exp26         = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']    = exp12 - exp26
    df['signal']  = df['macd'].ewm(span=9, adjust=False).mean()

    df['avg_vol'] = df['volume'].rolling(WINDOW).mean()
    return df.dropna()

# ─── Train or Load XGBoost Model ───────────────────────────────────────────────
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    df_train = fetch_data('SPY', '90 D')
    df_train = compute_indicators(df_train)
    df_train['future'] = df_train['close'].shift(-10)
    df_train['label']  = (df_train['future'] > df_train['close']).astype(int)
    
    features = df_train[['vwap', 'rsi', 'macd', 'signal', 'avg_vol']]
    labels   = df_train['label']
    X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size=0.2, shuffle=False)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print("XGB Accuracy:", accuracy_score(y_te, preds))
    joblib.dump(model, MODEL_FILE)

# ─── Backtest Simulation ────────────────────────────────────────────────────────
capital = INITIAL_CAP
logs    = []

for symbol in TICKERS:
    print(f"\n🔁 Backtesting {symbol}")
    df = fetch_data(symbol, '14 D')
    df = compute_indicators(df)

    position = None
    for i in range(WINDOW, len(df)):
        row       = df.iloc[i]
        price     = row['close']
        timestamp = row.name

        # Fakeout detection
        is_fakeout = (
            price > row['vwap']
            and df.iloc[i-1]['close'] > price
            and row['volume'] > 2 * row['avg_vol']
        )

        # Base entry condition
        base_cond = (
            price > row['vwap']
            and 30 < row['rsi'] < 70
            and row['macd'] > row['signal']
            and row['volume'] > 1.5 * row['avg_vol']
            and not is_fakeout
        )

        # AI filter via XGBoost
        feat    = np.array([[row['vwap'], row['rsi'], row['macd'], row['signal'], row['avg_vol']]])
        ai_cond = model.predict(feat)[0] == 1

        # ENTRY
        if position is None and base_cond and ai_cond:
            premium   = price * 0.02
            contracts = int((capital * ALLOC_PER_TRADE) / (premium * 100))
            position = {
                'symbol':      symbol,
                'entry_time':  timestamp,
                'entry_price': price,
                'contracts':   contracts
            }
            print(f"[ENTRY]  {timestamp} {symbol} | Price {price:.2f} | Contracts {contracts}")

        # EXIT
        elif position:
            price_move = price - position['entry_price']
            pnl        = price_move * DELTA * 100 * position['contracts']

            # Smarter exit: RSI divergence + MACD curl + volume drop
            smart_exit = (
                row['rsi'] < df.iloc[i-1]['rsi']
                and row['macd'] < df.iloc[i-1]['macd']
                and row['volume'] < row['avg_vol']
            )

            # Take-profit or smart exit
            if pnl >= INITIAL_CAP * ALLOC_PER_TRADE * 0.25 or smart_exit:
                capital += pnl
                logs.append({
                    'symbol':   symbol,
                    'entry':    position['entry_time'],
                    'exit':     timestamp,
                    'pnl':      pnl,
                    'duration': timestamp - position['entry_time'],
                    'type':     'TP/SmartExit'
                })
                print(f"[EXIT]   {timestamp} {symbol} | PnL {pnl:.2f}")
                position = None

            # Stop-loss at -20%
            elif pnl <= -INITIAL_CAP * ALLOC_PER_TRADE * 0.20:
                capital += pnl
                logs.append({
                    'symbol':   symbol,
                    'entry':    position['entry_time'],
                    'exit':     timestamp,
                    'pnl':      pnl,
                    'duration': timestamp - position['entry_time'],
                    'type':     'StopLoss'
                })
                print(f"[SL]     {timestamp} {symbol} | PnL {pnl:.2f}")
                position = None

    # Force-exit any open position at end
    if position:
        final_price = df.iloc[-1]['close']
        price_move  = final_price - position['entry_price']
        pnl         = price_move * DELTA * 100 * position['contracts']
        capital    += pnl
        logs.append({
            'symbol':   symbol,
            'entry':    position['entry_time'],
            'exit':     df.index[-1],
            'pnl':      pnl,
            'duration': df.index[-1] - position['entry_time'],
            'type':     'ForceExit'
        })
        print(f"[FORCE EXIT] {df.index[-1]} {symbol} | PnL {pnl:.2f}")
        position = None

# ─── Final Results & Logging ───────────────────────────────────────────────────
print(f"\n✅ Final Capital: ${capital:,.2f}")
ib.disconnect()
pd.DataFrame(logs).to_csv('milestone2_trade_log.csv', index=False)
print("📁 Trade log saved to milestone2_trade_log.csv")
