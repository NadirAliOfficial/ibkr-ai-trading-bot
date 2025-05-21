# milestone2_backtest_final.py

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
TICKERS         = ['SPY',  'AAPL']
INITIAL_CAP     = 100_000.0
ALLOC_PER_TRADE = 0.05
DELTA           = 0.5
WINDOW          = 20
MODEL_FILE      = 'xgb_odte_model.pkl'
TP_TIERS        = [1.25, 1.75, 2.50]   # +25%, +75%, +150%
SL_THRESHOLD    = 0.20                # 20% stop-loss

# ─── Connect ───────────────────────────────────────────────────────────────────
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)

def fetch_data(symbol, duration='14 D'):
    contract = Stock(symbol, 'SMART', 'USD', primaryExchange='ARCA')
    ib.qualifyContracts(contract)
    for _ in range(3):
        bars = ib.reqHistoricalData(contract, '', duration, '1 min', 'TRADES', True)
        if bars:
            return util.df(bars).set_index('date')
        time.sleep(2)
    raise RuntimeError(f"Failed fetching {symbol}")

def compute_indicators(df):
    df['vwap']    = (df['close']*df['volume']).rolling(WINDOW).sum() / df['volume'].rolling(WINDOW).sum()
    d            = df['close'].diff()
    gain         = d.clip(lower=0)
    loss         = -d.clip(upper=0)
    avg_gain     = gain.rolling(14).mean()
    avg_loss     = loss.rolling(14).mean()
    rs           = avg_gain/(avg_loss+1e-5)
    df['rsi']    = 100-(100/(1+rs))
    e12          = df['close'].ewm(span=12,adjust=False).mean()
    e26          = df['close'].ewm(span=26,adjust=False).mean()
    df['macd']   = e12-e26
    df['signal'] = df['macd'].ewm(span=9,adjust=False).mean()
    df['avg_vol']= df['volume'].rolling(WINDOW).mean()
    return df.dropna()

# ─── Load or train XGBoost ───────────────────────────────────────────────────
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    train_df = fetch_data('SPY', '90 D')
    train_df = compute_indicators(train_df)
    train_df['future'] = train_df['close'].shift(-10)
    train_df['label']  = (train_df['future'] > train_df['close']).astype(int)
    feats = train_df[['vwap','rsi','macd','signal','avg_vol']]
    labels= train_df['label']
    Xtr,Xte,ytr,yte = train_test_split(feats,labels,test_size=0.2,shuffle=False)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3,
                              use_label_encoder=False, eval_metric='logloss')
    model.fit(Xtr,ytr)
    print("XGB Acc:", accuracy_score(yte, model.predict(Xte)))
    joblib.dump(model, MODEL_FILE)

capital    = INITIAL_CAP
tickerPL   = {t: 0.0 for t in TICKERS}
all_logs   = []

for sym in TICKERS:
    print(f"\n🔁 Backtesting {sym}")
    df = fetch_data(sym, '14 D')
    df = compute_indicators(df)

    position  = None
    targets   = []
    ti        = 0

    for i in range(WINDOW, len(df)):
        row = df.iloc[i]
        p   = row['close']
        t   = row.name

        # Fakeout
        fake = (p>row['vwap'] and df.iloc[i-1]['close']>p and row['volume']>2*row['avg_vol'])

        base = (p>row['vwap'] and 30<row['rsi']<70 and
                row['macd']>row['signal'] and row['volume']>1.5*row['avg_vol'] and not fake)
        feat = np.array([[row['vwap'],row['rsi'],row['macd'],row['signal'],row['avg_vol']]])
        aiok = model.predict(feat)[0]==1

        if position is None and base and aiok:
            prem     = p*0.02
            cnt      = int((capital*ALLOC_PER_TRADE)/(prem*100))
            position = {'entry_t':t,'entry_p':p,'cnt':cnt}
            targets  = [p*x for x in TP_TIERS]
            ti       = 0
            print(f"[ENTRY]  {t} {sym} @ {p:.2f} → {cnt} contracts")

        elif position:
            move   = p-position['entry_p']
            pnl    = move*DELTA*100*position['cnt']

            # Tier exits
            if ti<len(targets) and p>=targets[ti]:
                slice_qty = position['cnt']//len(targets)
                gain = (targets[ti]-position['entry_p'])*DELTA*100*slice_qty
                capital += gain
                tickerPL[sym]+= gain
                all_logs.append({'sym':sym,'entry':position['entry_t'],
                                 'exit':t,'pnl':gain,'type':f'Tier{ti+1}'})
                print(f"[TIER{ti+1}] {t} {sym} PnL ${gain:.2f}")
                ti+=1
                if ti>=len(targets):
                    position=None

            # Stop-loss
            elif pnl<=-INITIAL_CAP*ALLOC_PER_TRADE*SL_THRESHOLD:
                capital += pnl
                tickerPL[sym]+= pnl
                all_logs.append({'sym':sym,'entry':position['entry_t'],
                                 'exit':t,'pnl':pnl,'type':'SL'})
                print(f"[SL]     {t} {sym} PnL ${pnl:.2f}")
                position=None

            # Smart exit
            elif (row['rsi']<df.iloc[i-1]['rsi'] and
                  row['macd']<df.iloc[i-1]['macd'] and
                  row['volume']<row['avg_vol']):
                capital += pnl
                tickerPL[sym]+= pnl
                all_logs.append({'sym':sym,'entry':position['entry_t'],
                                 'exit':t,'pnl':pnl,'type':'SmartExit'})
                print(f"[SmartExit] {t} {sym} PnL ${pnl:.2f}")
                position=None

# ─── Reporting ────────────────────────────────────────────────────────────────
print("\n📈 Ticker PnL Breakdown:")
for s,pl in tickerPL.items():
    print(f"  {s}: ${pl:.2f}")
total_profit = capital - INITIAL_CAP
print(f"\n💰 Total Profit: ${total_profit:.2f}")
print(f"✅ Final Capital: ${capital:.2f}")

pd.DataFrame(all_logs).to_csv('milestone2_trade_log.csv', index=False)
print("📁 Logs → milestone2_trade_log.csv")

ib.disconnect()
