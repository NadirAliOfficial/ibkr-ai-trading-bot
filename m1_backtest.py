# m1_backtest.py

from ib_insync import IB, Stock, util
import pandas as pd
import numpy as np

# ─── Connect to IBKR Paper Account and fetch data ───────────────────────────────
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)   # TWS/IB Gateway must be running

contract = Stock('SPY', 'SMART', 'USD', primaryExchange='ARCA')
ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='14 D',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=True,
)
df = util.df(bars)
df.set_index('date', inplace=True)

# ─── Indicator Calculation ─────────────────────────────────────────────────────
WINDOW = 20

# VWAP (rolling)
df['vwap'] = (
    (df['close'] * df['volume']).rolling(WINDOW).sum() /
    df['volume'].rolling(WINDOW).sum()
)

# RSI (14)
delta     = df['close'].diff()
gain      = delta.clip(lower=0)
loss      = -delta.clip(upper=0)
avg_gain  = gain.rolling(14).mean()
avg_loss  = loss.rolling(14).mean()
rs        = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

# MACD & Signal
exp12           = df['close'].ewm(span=12, adjust=False).mean()
exp26           = df['close'].ewm(span=26, adjust=False).mean()
df['macd']      = exp12 - exp26
df['signal']    = df['macd'].ewm(span=9, adjust=False).mean()

# Volume spike detection
df['avg_vol']   = df['volume'].rolling(WINDOW).mean()

# ─── Backtest Simulation ───────────────────────────────────────────────────────
capital   = 100_000.00
position  = None

for i in range(WINDOW, len(df)):
    row = df.iloc[i]
    price = row['close']

    # ENTRY: no open position
    if position is None:
        cond = (
            price > row['vwap'] and
            30 < row['rsi'] < 70 and
            row['macd'] > row['signal'] and
            row['volume'] > 1.5 * row['avg_vol']
        )
        if cond:
            qty = int((capital * 0.05) / price)  # 5% allocation
            if qty > 0:
                entry_price = price
                position = {
                    'qty_total':     qty,
                    'qty_remaining': qty,
                    'entry':         entry_price,
                    'targets':       [
                        entry_price * 1.25,
                        entry_price * 1.75,
                        entry_price * 2.50
                    ],
                    'tgt_idx':       0,
                    'stop':          entry_price * 0.98
                }
                print(f"[ENTRY] {row.name} — Buy {qty} @ ${entry_price:.2f}")

    # MANAGEMENT: open position exists
    else:
        # stop‐loss: sell all
        if price <= position['stop']:
            qty = position['qty_remaining']
            pnl = qty * (price - position['entry'])
            capital += pnl
            print(f"[STOP]  {row.name} — Sell {qty} @ ${price:.2f} | PnL: {pnl:.2f}")
            position = None

        # tiered profit targets
        else:
            tgt_price = position['targets'][position['tgt_idx']]
            if price >= tgt_price:
                slice_qty = position['qty_total'] // len(position['targets'])
                pnl = slice_qty * (tgt_price - position['entry'])
                capital += pnl
                position['qty_remaining'] -= slice_qty
                print(f"[TAKE]  {row.name} — Sell {slice_qty} @ ${tgt_price:.2f} | PnL: {pnl:.2f}")
                position['tgt_idx'] += 1

                # all slices done
                if position['tgt_idx'] >= len(position['targets']):
                    position = None

# ─── Force‐exit any open position to capture unrealized PnL ─────────────────────
if position is not None:
    final_price = df.iloc[-1]['close']
    qty         = position['qty_remaining']
    pnl         = qty * (final_price - position['entry'])
    capital    += pnl
    print(f"[FORCE EXIT] {df.index[-1]} — Sell {qty} @ ${final_price:.2f} | PnL: {pnl:.2f}")

# ─── Results ───────────────────────────────────────────────────────────────────
print(f"\nFinal capital: ${capital:,.2f}")
ib.disconnect()
