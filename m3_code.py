from tkinter import *
from threading import Thread
import asyncio
import pandas as pd
import numpy as np
import time
import os
import joblib

from ib_insync import IB, Stock, Option, util
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ─── Model Paths ─────────────────────────────────────────────────────────────
PICKLE_MODEL = "/Users/nadirali/code/ibkr-ai-trading-bot/xgb_odte_model.pkl"
JSON_MODEL   = "/Users/nadirali/code/ibkr-ai-trading-bot/xgb_odte_model.json"

# ─── GUI Setup ───────────────────────────────────────────────────────────────
root = Tk()
root.title("IBKR Backtest GUI (Real Options)")
root.geometry("750x650")

Label(root, text="Initial Capital").grid(row=0, column=0, sticky=W, padx=10)
initial_cap_entry = Entry(root); initial_cap_entry.insert(0, "100000")
initial_cap_entry.grid(row=0, column=1, padx=5, pady=5)

Label(root, text="Tickers (comma-separated)").grid(row=1, column=0, sticky=W, padx=10)
tickers_entry = Entry(root); tickers_entry.insert(0, "SPY,AAPL")
tickers_entry.grid(row=1, column=1, padx=5, pady=5)

Label(root, text="Allocation Per Trade (%)").grid(row=2, column=0, sticky=W, padx=10)
alloc_entry = Entry(root); alloc_entry.insert(0, "10")
alloc_entry.grid(row=2, column=1, padx=5, pady=5)

Label(root, text="Stop-loss Threshold (%)").grid(row=3, column=0, sticky=W, padx=10)
sl_entry = Entry(root); sl_entry.insert(0, "5")
sl_entry.grid(row=3, column=1, padx=5, pady=5)

Label(root, text="Option Expiry (YYYYMMDD)").grid(row=4, column=0, sticky=W, padx=10)
expiry_entry = Entry(root); expiry_entry.insert(0, "20240620")
expiry_entry.grid(row=4, column=1, padx=5, pady=5)

Label(root, text="Strike Offset ($)").grid(row=5, column=0, sticky=W, padx=10)
offset_entry = Entry(root); offset_entry.insert(0, "5")
offset_entry.grid(row=5, column=1, padx=5, pady=5)

output_text = Text(root, height=25, width=90, state='disabled', bg="#f0f0f0")

def log(msg):
    output_text.configure(state='normal')
    output_text.insert(END, msg + "\n")
    output_text.see(END)
    output_text.configure(state='disabled')

# ─── Backtest Logic ──────────────────────────────────────────────────────────
def run_backtest():
    asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        # read GUI inputs
        initial_cap   = float(initial_cap_entry.get())
        tickers       = [t.strip().upper() for t in tickers_entry.get().split(",")]
        alloc_pct     = float(alloc_entry.get()) / 100
        sl_thresh     = float(sl_entry.get()) / 100
        expiry        = expiry_entry.get()
        strike_offset = float(offset_entry.get())

        # connect to IBKR
        ib = IB()
        try:
            
            ib.connect('127.0.0.1', 7497, clientId=2, readonly=True)
            ib.reqMarketDataType(1)  # 1 = real-time data
            time.sleep(1)

        except Exception as e:
            log(f"❌ IB connect error: {e}")
            return

        # helper: fetch 1-min bars, with retry
        def fetch_data(sym, duration='14 D'):
            contract = Stock(sym, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            for _ in range(3):

                bars = ib.reqHistoricalData(
                        contract, '', duration, '1 min', 'TRADES', useRTH=False
                    )

                if bars:
                    df = util.df(bars).set_index('date')
                    if not df.empty:
                        return df
                time.sleep(2)
            raise RuntimeError(f"No data for {sym}")

        # helper: indicators
        def compute_indicators(df):
            df['vwap']    = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            d             = df['close'].diff()
            gain, loss    = d.clip(lower=0), -d.clip(upper=0)
            df['rsi']     = 100 - (100 / (1 + gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-5)))
            e12, e26      = df['close'].ewm(span=12, adjust=False).mean(), df['close'].ewm(span=26, adjust=False).mean()
            df['macd']    = e12 - e26
            df['signal']  = df['macd'].ewm(span=9, adjust=False).mean()
            df['avg_vol'] = df['volume'].rolling(20).mean()
            return df.dropna()

        # helper: real option premium
        def get_real_option_premium(sym, expiry, offset, right='C'):
            # get underlying price
            stk = Stock(sym, 'SMART', 'USD')
            ib.qualifyContracts(stk)
            ib.sleep(1.5)  # let IBKR warm up after contract qualification
            tck = ib.reqMktData(stk, '', False, False)
            time.sleep(1)
            price = tck.last
            ib.cancelMktData(stk)
            if price is None or np.isnan(price):
                return None
            strike = round(price + offset) if right == 'C' else round(price - offset)

            opt = Option(sym, expiry, strike, right, 'SMART')
            ib.qualifyContracts(opt)
            bars = ib.reqHistoricalData(
                opt, '', '1 D', '1 min', 'MIDPOINT', useRTH=False
            )

            if not bars:
                return None
            df_opt = util.df(bars)
            prem = df_opt['close'].iloc[-1]
            if pd.isna(prem) or prem <= 0:
                return None
            return float(prem)

        # load/convert/train model
        if os.path.exists(JSON_MODEL):
            model = xgb.XGBClassifier()
            model.load_model(JSON_MODEL)
            log(f"✅ Loaded JSON model")
        else:
            if os.path.exists(PICKLE_MODEL):
                model = joblib.load(PICKLE_MODEL)
                model.save_model(JSON_MODEL)
                log("✅ Converted pickle → JSON")
            else:
                # train from SPY
                df_train = fetch_data('SPY', '90 D')
                df_train = compute_indicators(df_train)
                df_train['future'] = df_train['close'].shift(-10)
                df_train.dropna(inplace=True)  # drop NaNs from shift
                df_train['label']  = (df_train['future'] > df_train['close']).astype(int)
                feats, labels = df_train[['vwap','rsi','macd','signal','avg_vol']], df_train['label']
                Xtr, Xte, ytr, yte = train_test_split(feats, labels, test_size=0.2, shuffle=False)
                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=3,
                    use_label_encoder=False, eval_metric='logloss'
                )
                model.fit(Xtr, ytr)
                model.save_model(JSON_MODEL)
                log("✅ Trained new JSON model")

        capital = initial_cap
        TP_TIERS = [1.25, 1.75, 2.50]
        DELTA    = 0.5

        # loop tickers
        for sym in tickers:
            try:
                df = fetch_data(sym, '14 D')
            except Exception as e:
                log(f"⚠️ Skipping {sym}: {e}")
                continue

            df = compute_indicators(df)
            log(f"\n🔁 Backtesting {sym}")

            position = None
            tier_i   = 0

            for i in range(20, len(df)):
                row = df.iloc[i]
                p, t = row['close'], row.name

                fake = (
                    p > row['vwap'] and
                    df.iloc[i-1]['close'] > p and
                    row['volume'] > 2 * row['avg_vol']
                )
                base = (
                    p > row['vwap'] and
                    30 < row['rsi'] < 70 and
                    row['macd'] > row['signal'] and
                    row['volume'] > 1.5 * row['avg_vol'] and
                    not fake
                )
                feat = np.array([[row['vwap'], row['rsi'], row['macd'], row['signal'], row['avg_vol']]])
                aiok = model.predict(feat)[0] == 1

                if position is None and base and aiok:
                    opt_prem = get_real_option_premium(sym, expiry, strike_offset, 'C')
                    if opt_prem is None:
                        log(f"⚠️ No option premium for {sym} at {t}, skipping")
                        continue

                    cnt = int((capital * alloc_pct) / (opt_prem * 100))
                    position = {'entry_t': t, 'entry_p': p, 'cnt': cnt}
                    targets = [p * x for x in TP_TIERS]
                    tier_i   = 0
                    log(f"[ENTRY] {t} {sym} @ {p:.2f} | contracts={cnt} | prem={opt_prem:.2f}")

                elif position:
                    move = p - position['entry_p']
                    pnl  = move * DELTA * 100 * position['cnt']

                    # tiered profit
                    if tier_i < len(TP_TIERS) and p >= targets[tier_i]:
                        gain = (targets[tier_i] - position['entry_p']) * DELTA * 100 * (position['cnt'] // len(targets))
                        capital += gain
                        log(f"[TIER{tier_i+1}] {t} PnL=${gain:.2f}")
                        tier_i += 1
                        if tier_i == len(TP_TIERS):
                            position = None

                    # stop-loss
                    elif pnl <= -initial_cap * alloc_pct * sl_thresh:
                        capital += pnl
                        log(f"[SL] {t} PnL=${pnl:.2f}")
                        position = None

                    # smart exit
                    elif (
                        row['rsi']  < df.iloc[i-1]['rsi'] and
                        row['macd'] < df.iloc[i-1]['macd'] and
                        row['volume'] < row['avg_vol']
                    ):
                        capital += pnl
                        log(f"[SmartExit] {t} PnL=${pnl:.2f}")
                        position = None

        log(f"\n💰 Final Capital: ${capital:.2f}")
        ib.disconnect()

    except Exception as e:
        log(f"❌ Error: {e}")

# ─── Button & Log ────────────────────────────────────────────────────────────
Button(
    root,
    text="Start Backtest",
    command=lambda: Thread(target=run_backtest).start(),
    bg="#4CAF50", fg="white", width=20
).grid(row=6, column=0, columnspan=2, pady=10)

output_text.grid(row=7, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()
