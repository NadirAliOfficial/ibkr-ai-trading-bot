from ib_insync import *
import pandas as pd
import numpy as np

# Connect to IBKR Paper Account
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Ensure TWS or IB Gateway is running

# Define SPY contract
contract = Stock('SPY', 'SMART', 'USD')

# Request historical data (1-minute bars for 1 day)
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1
)

# Convert to DataFrame
df = util.df(bars)
df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
df['rsi'] = df['close'].rolling(14).apply(
    lambda x: 100 - (100 / (1 + np.mean(np.diff(x)[np.diff(x) > 0]) /
                            np.abs(np.mean(np.diff(x)[np.diff(x) <= 0]) + 1e-5)))
)

# Simulate strategy: Buy if price > VWAP and RSI < 30
entry_price = None
for i in range(15, len(df)):
    row = df.iloc[i]
    if entry_price is None:
        if row['close'] > row['vwap'] and row['rsi'] < 30:
            entry_price = row['close']
            entry_time = row['date']
            print(f"[ENTRY] {entry_time} — Buy at ${entry_price:.2f}")
    else:
        gain = (row['close'] - entry_price) / entry_price
        if gain >= 0.02 or gain <= -0.01:  # Simulated TP/SL
            print(f"[EXIT]  {row['date']} — Exit at ${row['close']:.2f} | PnL: {gain*100:.2f}%")
            entry_price = None

ib.disconnect()
