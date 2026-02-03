import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime

TICKERS = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "GOOGL", "AMZN", "META", "PLTR", "SPY"]
LOG_FILE = 'final_backtest_results.csv'
SIGNAL_FILE = 'ibkr_signals.json'

def run_agent():
    print(f"--- START AGENTA: {datetime.now()} ---")
    
    signals = {"Strategy_A": [], "Strategy_B": [], "Strategy_V": []}
    today_trades_log = []
    
    for t in TICKERS:
        try:
            print(f"Stahuji data pro {t}...")
            # Stahujeme 40 dní historie (pro indikátory) a dnešní 1m data (pro vyhodnocení)
            d = yf.download(t, period="40d", interval="1d", progress=False)
            
            if d.empty or len(d) < 21:
                print(f"   ⚠️ Málo dat pro {t}, přeskakuji.")
                continue

            # Extrakce hodnot (zajištění, že bereme skaláry)
            # Poslední řádek je index -1 (dnešek/poslední close), předchozí je -2
            prev_high20 = float(d['High'].rolling(window=20).max().iloc[-2])
            prev_low20 = float(d['Low'].rolling(window=20).min().iloc[-2])
            
            curr_row = d.iloc[-1]
            c_close = float(curr_row['Close'])
            c_high = float(curr_row['High'])
            c_low = float(curr_row['Low'])
            c_open = float(curr_row['Open'])
            c_vol = float(curr_row['Volume'])
            
            prev_row = d.iloc[-2]
            p_range = float(prev_row['High'] - prev_row['Low'])
            avg_range = float((d['High'] - d['Low']).rolling(window=20).mean().iloc[-2])
            avg_vol = float(d['Volume'].rolling(window=20).mean().iloc[-2])

            # --- LOGIKA SIGNÁLŮ ---
            # Strat A (Setup)
            dist_h = abs(c_close - prev_high20) / (avg_range + 1e-9)
            if dist_h < 0.4:
                signals["Strategy_A"].append({'ticker': t, 'action': 'BUY', 'score': 1/(dist_h+0.01), 'range': p_range, 'price': c_close})

            # Strat V (Reactive)
            if c_high > prev_high20:
                signals["Strategy_V"].append({'ticker': t, 'action': 'BUY', 'score': p_range, 'range': p_range, 'price': c_close})

            # --- OKAMŽITÉ VYHODNOCENÍ DNEŠKA ---
            # Simulujeme obchod za dnešní Open -> Close
            pnl = (int(10000/c_open) * (c_close - c_open)) - 1.0
            today_trades_log.append([datetime.now().strftime('%Y-%m-%d'), "DailyAgent", t, "Long", round(c_open, 2), round(c_close, 2), "EOD", round(pnl, 2)])

        except Exception as e:
            print(f"   ❌ Chyba u {t}: {e}")

    # Uložení signálů
    final_json = {}
    for m in signals:
        top3 = sorted(signals[m], key=lambda x: x['score'], reverse=True)[:3]
        final_json[m] = top3
    
    with open(SIGNAL_FILE, 'w') as f:
        json.dump(final_json, f, indent=4)
    print(f"✅ Signály uloženy do {SIGNAL_FILE}")

    # Uložení výsledků do CSV
    if today_trades_log:
        df_new = pd.DataFrame(today_trades_log, columns=['Date', 'Strategy', 'Ticker', 'Side', 'Entry', 'Exit', 'Type', 'Profit'])
        if not os.path.exists(LOG_FILE):
            df_new.to_csv(LOG_FILE, index=False)
        else:
            df_new.to_csv(LOG_FILE, mode='a', header=False, index=False)
        print(f"✅ Výsledky dopsány do {LOG_FILE}")

if __name__ == "__main__":
    run_agent()
