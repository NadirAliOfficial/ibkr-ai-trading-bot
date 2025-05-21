import streamlit as st
import nest_asyncio
nest_asyncio.apply()
import json, os, time, math
import pandas as pd
import numpy as np
from datetime import datetime
from ib_insync import IB, Stock, Option, util
from scipy.stats import norm

# ─── Load or Create Config ─────────────────────────────────────────────────────
CONFIG_FILE = 'config.json'
if os.path.exists(CONFIG_FILE):
    cfg = json.load(open(CONFIG_FILE))
else:
    st.error("Missing strategy_config.json")
    st.stop()

# ─── Sidebar: Strategy Parameters ───────────────────────────────────────────────
st.sidebar.title("⚙️ Strategy Settings")
cfg['allocation_per_trade'] = st.sidebar.slider("Allocation %", 1, 20, int(cfg['allocation_per_trade']*100)) / 100.0
cfg['vwap_window']         = st.sidebar.number_input("VWAP Window", 5, 60, cfg['vwap_window'])
cfg['rsi_window']          = st.sidebar.number_input("RSI Window", 5, 30, cfg['rsi_window'])
cfg['macd_fast']           = st.sidebar.number_input("MACD Fast EMA", 5, 30, cfg['macd_fast'])
cfg['macd_slow']           = st.sidebar.number_input("MACD Slow EMA", 10, 60, cfg['macd_slow'])
cfg['macd_signal']         = st.sidebar.number_input("MACD Signal EMA", 5, 30, cfg['macd_signal'])
cfg['volume_spike_mult']   = st.sidebar.number_input("Volume Spike Mult", 1.0, 5.0, cfg['volume_spike_mult'], 0.1)
tiers_str                  = st.sidebar.text_input("Profit Tiers % (comma)", ",".join(str(int(t*100)) for t in cfg['tp_tiers']))
cfg['tp_tiers']            = [int(x)/100 for x in tiers_str.split(",") if x.strip()]
cfg['sl_pct']              = st.sidebar.slider("Stop-Loss %", 5, 50, int(cfg['sl_pct']*100)) / 100.0
cfg['use_black_scholes']   = st.sidebar.checkbox("Black-Scholes Pricing", cfg['use_black_scholes'])

if st.sidebar.button("Save Config"):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)
    st.sidebar.success("Saved!")

# ─── Main: Backtest Controls ────────────────────────────────────────────────────
st.title("📈 0DTE Options Backtest & Config")

ticker = st.selectbox("Ticker to Backtest", cfg['tickers'])
if st.button("Run Backtest"):
    # ─── Connect to IBKR (Paper Mode) ───────────────────────────
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=10)
    # ─── Fetch 14-day 1-min bars ────────────────────────────────
    bars = ib.reqHistoricalData(
        Stock(ticker,'SMART','USD'), '', '14 D', '1 min', 'TRADES', True
    )
    df = util.df(bars).set_index('date')

    # ─── Compute Indicators ───────────────────────────────────
    # VWAP
    df['vwap'] = (df['close']*df['volume']).rolling(cfg['vwap_window']).sum() / \
                  df['volume'].rolling(cfg['vwap_window']).sum()
    # RSI
    d       = df['close'].diff()
    gain    = d.clip(lower=0).rolling(cfg['rsi_window']).mean()
    loss    = -d.clip(upper=0).rolling(cfg['rsi_window']).mean()
    df['rsi'] = 100 - (100/(1+gain/(loss+1e-5)))
    # MACD + signal
    expf      = df['close'].ewm(span=cfg['macd_fast'], adjust=False).mean()
    exps      = df['close'].ewm(span=cfg['macd_slow'], adjust=False).mean()
    df['macd'] = expf - exps
    df['signal'] = df['macd'].ewm(span=cfg['macd_signal'], adjust=False).mean()
    df['avg_vol'] = df['volume'].rolling(cfg['vwap_window']).mean()

    # ─── Backtest Loop ─────────────────────────────────────────
    capital = 100_000.0
    logs    = []
    position = None

    for i in range(cfg['vwap_window'], len(df)):
        row   = df.iloc[i]
        price = row['close']
        now   = row.name

        # Entry filters
        cond = (
            price > row['vwap'] and
            row['volume'] > cfg['volume_spike_mult']*row['avg_vol'] and
            30 < row['rsi'] < 70 and
            row['macd'] > row['signal']
        )
        if position is None and cond:
            # simulate 0DTE ATM contract premium
            premium = price * cfg['delta_target'] * 0.02
            qty     = int((capital*cfg['allocation_per_trade'])/(premium*100))
            if qty>0:
                position = dict(entry_p=price, qty=qty, premium=premium, 
                                targets=[price*(1+t) for t in cfg['tp_tiers']], ti=0, t0=i)
                logs.append({'time':now,'type':'ENTRY','price':price,'qty':qty})
        elif position:
            move = price-position['entry_p']
            pnl  = move*100*position['qty']
            # Tier exit
            if position['ti']<len(position['targets']) and price>=position['targets'][position['ti']]:
                logs.append({'time':now,'type':f'TIER{position["ti"]+1}','pnl':pnl})
                capital += pnl
                position['ti']+=1
                if position['ti']==len(position['targets']):
                    position=None
            # Stop-loss
            elif pnl<=-cfg['sl_pct']*position['premium']*100*position['qty']:
                logs.append({'time':now,'type':'SL','pnl':pnl})
                capital += pnl
                position=None

    ib.disconnect()

    # ─── Results & Visualization ───────────────────────────────
    df_logs = pd.DataFrame(logs).set_index('time')
    df_logs['cumPnL'] = df_logs['pnl'].fillna(0).cumsum()

    st.subheader("💹 Cumulative PnL")
    st.line_chart(df_logs['cumPnL'])

    st.subheader("📋 Trade Log")
    st.dataframe(df_logs)

    st.subheader("🏁 Final Capital")
    st.metric("Final Capital", f"${capital:,.2f}")
