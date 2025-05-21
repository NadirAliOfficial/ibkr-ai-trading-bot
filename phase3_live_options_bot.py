import os
import json
import time
import math
import logging
from datetime import datetime
from ib_insync import IB, Option, util, Stock
import numpy as np
import pandas as pd
from scipy.stats import norm

# ─── Load Strategy Config ───────────────────────────────────────────────────────
CONFIG_FILE = 'strategy_config.json'
if not os.path.exists(CONFIG_FILE):
    default_cfg = {
        "tickers": ["SPY","QQQ","AAPL","TSLA"],
        "allocation_per_trade": 0.05,
        "rsi_window": 14,
        "macd": {"fast":12,"slow":26,"signal":9},
        "vwap_window": 20,
        "volume_spike_mult": 1.5,
        "tp_tiers": [0.25,0.75,1.5],
        "sl_pct": 0.20,
        "delta_target": 0.5,
        "time_filters": [["09:40","10:00"],["15:15","15:45"]],
        "use_black_scholes": True,
        "risk_free_rate": 0.02
    }
    with open(CONFIG_FILE,'w') as f:
        json.dump(default_cfg, f, indent=2)

with open(CONFIG_FILE) as f:
    cfg = json.load(f)

# ─── Black-Scholes Pricing ──────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, is_call=True):
    if T <= 0: 
        return max(0.0, (S-K) if is_call else (K-S))
    d1 = (math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if is_call:
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# ─── IBKR Connection ────────────────────────────────────────────────────────────
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=3)
logging.basicConfig(level=logging.INFO)

# ─── Helpers ────────────────────────────────────────────────────────────────────
def fetch_underlying(symbol, duration='1 D'):
    """Fetch 1-min bars for underlying."""
    bars = ib.reqHistoricalData(Stock(symbol,'SMART','USD'), '', duration, '1 min', 'TRADES', True)
    return util.df(bars).set_index('date')

def fetch_0dte_strike(symbol):
    """Fetch today’s ATM strike."""
    exp = datetime.now().strftime('%Y%m%d')
    chains = ib.reqSecDefOptParams(symbol, '', 'SMART', Stock(symbol,'SMART','USD'))
    chain = next((c for c in chains if exp in c.expirations and c.tradingClass==symbol), None)
    if not chain:
        raise RuntimeError("No 0DTE chain found")
    underlying = ib.reqTickers(Stock(symbol,'SMART','USD'))[0].marketPrice()
    return min(chain.strikes, key=lambda k: abs(k-underlying))

def make_option(symbol, strike, is_call=True):
    return Option(symbol, strike, datetime.now().strftime('%Y%m%d'),
                  'C' if is_call else 'P', 'SMART')

# ─── Main Loop ─────────────────────────────────────────────────────────────────
capital = 100_000.0

while True:
    now = datetime.now().strftime('%H:%M')
    # time filters
    if any(start <= now <= end for start,end in cfg['time_filters']):
        time.sleep(60)
        continue

    for sym in cfg['tickers']:
        df = fetch_underlying(sym)
        df['vwap'] = (df['close']*df['volume']).rolling(cfg['vwap_window']).sum() / df['volume'].rolling(cfg['vwap_window']).sum()
        # (compute RSI, MACD, volume spike similarly)
        latest = df.iloc[-1]
        price  = latest['close']

        # simple entry check example
        if not (price > latest['vwap']): 
            continue

        strike = fetch_0dte_strike(sym)
        opt = make_option(sym, strike, True)
        md  = ib.reqTickers(opt)[0]

        if cfg['use_black_scholes']:
            sigma = df['close'].pct_change().rolling(cfg['vwap_window']).std().iloc[-1]*math.sqrt(252)
            T = max((datetime.now().replace(hour=16,minute=0)-datetime.now()).seconds/86400, 0)
            theo = bs_price(price, strike, T, cfg['risk_free_rate'], sigma, True)
            exec_price = (theo + md.ask)/2
        else:
            exec_price = (md.bid + md.ask)/2

        qty = int((capital*cfg['allocation_per_trade'])/(exec_price*100))
        if qty<=0:
            continue

        logging.info(f"BUY {sym} {strike}C x{qty} @ {exec_price:.2f}")
        ib.placeOrder(opt, ib.LimitOrder('BUY', qty, exec_price))

    time.sleep(60)
