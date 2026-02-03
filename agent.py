import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime

# --- KONFIGURACE ---
TICKERS = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "GOOGL", "AMZN", "META", "PLTR", "SPY", "QQQ"]
LOG_FILE = 'final_backtest_results.csv'
SIGNAL_FILE = 'ibkr_signals.json'

def calculate_z_score(profits):
    if len(profits) < 5: return 0 # Potřebujeme aspoň 5 vzorků pro relevanci
    mean = np.mean(profits)
    std = np.std(profits)
    return mean / (std + 1e-9)

def run_agent():
    print(f"🚀 Start agenta: {datetime.now()}")
    
    # 1. STAŽENÍ DAT S OŠETŘENÍM CHYB
    try:
        raw = yf.download(TICKERS, period="2y", interval="1d", group_by='ticker', progress=False)
        if raw.empty:
            print("❌ Selhalo stahování všech dat z yfinance.")
            return
    except Exception as e:
        print(f"❌ Kritická chyba při komunikaci s API: {e}")
        return
    
    ticker_data = {}
    for t in TICKERS:
        try:
            # Kontrola, zda ticker v datech vůbec je a není prázdný
            if t not in raw or raw[t].dropna().empty:
                print(f"⚠️ Ticker {t} nemá dostupná data, přeskakuji.")
                continue
                
            d = raw[t].dropna().copy()
            # OPRAVA: Shift(2) pro indikátory (vstupujeme na Open dne T, známe data T-1)
            d['Prev_High20_Strict'] = d['High'].rolling(window=20).max().shift(2)
            d['Prev_Low20_Strict'] = d['Low'].rolling(window=20).min().shift(2)
            d['Prev_Range'] = (d['High'] - d['Low']).shift(1)
            d['Prev_AvgRange'] = d['Prev_Range'].rolling(window=20).mean()
            d['Prev_Close'] = d['Close'].shift(1)
            d['Prev_Open'] = d['Open'].shift(1)
            d['Prev_Volume'] = d['Volume'].shift(1)
            d['Prev_V_Avg'] = d['Volume'].rolling(window=20).mean().shift(1)
            d['Day_Return_Pct'] = (d['Close'] - d['Open']) / d['Open']
            
            ticker_data[t] = d.dropna()
        except Exception as e:
            print(f"⚠️ Chyba při zpracování {t}: {e}")

    # 2. HISTORICKÉ Z-SCORE (Backtest na historii tickeru)
    ticker_performance = {m: {} for m in ['A', 'B', 'V']}
    
    for t, df in ticker_data.items():
        hist_df = df.iloc[:-5] # In-Sample (bez posledních dnů)
        
        # Strat A: Setup u High20
        sig_a = hist_df[abs(hist_df['Prev_Close'] - hist_df['Prev_High20_Strict']) / (hist_df['Prev_AvgRange'] + 1e-9) < 0.4]
        ticker_performance['A'][t] = calculate_z_score(sig_a['Day_Return_Pct'])

        # Strat B: Volume spike (zjednodušeno na Long pro skórování)
        sig_b = hist_df[hist_df['Prev_Volume'] > hist_df['Prev_V_Avg'] * 1.5]
        ticker_performance['B'][t] = calculate_z_score(sig_b['Day_Return_Pct'])

        # Strat V: Breakout (OPRAVENO: Porovnáváme High dne T-1 s High20 platným pro ten den)
        sig_v = hist_df[hist_df['High'].shift(1) > hist_df['Prev_High20_Strict']]
        ticker_performance['V'][t] = calculate_z_score(sig_v['Day_Return_Pct'])

    # 3. GENERACE SIGNÁLŮ (Dnešní řádek)
    final_signals = {}
    eval_logs = []
    
    for mode in ['A', 'B', 'V']:
        candidates = []
        for t, df in ticker_data.items():
            row = df.iloc[-1]
            is_signal = False
            side = 'Long'
            
            if mode == 'A':
                dist_h = abs(row['Prev_Close'] - row['Prev_High20_Strict']) / (row['Prev_AvgRange'] + 1e-9)
                if dist_h < 0.4: 
                    is_signal = True; side = 'Long'
            elif mode == 'B':
                if row['Prev_Volume'] > row['Prev_V_Avg'] * 1.5:
                    is_signal = True
                    side = 'Long' if row['Prev_Close'] > row['Prev_Open'] else 'Short'
            elif mode == 'V':
                # OPRAVA: Odstraněno .iloc[-1], porovnáváme přímo skalár z 'row'
                if row['High'] > row['Prev_High20_Strict']:
                    is_signal = True; side = 'Long'

            if is_signal:
                z_score = ticker_performance[mode].get(t, 0)
                candidates.append({'ticker': t, 'side': side, 'score': z_score})
        
        sel = sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
        final_signals[mode] = sel
        
        for s in sel:
            d_row = ticker_data[s['ticker']].iloc[-1]
            pnl = (int(10000/d_row['Open']) * (d_row['Close'] - d_row['Open'] if s['side'] == 'Long' else d_row['Open'] - d_row['Close'])) - 1.0
            eval_logs.append({
                'Date': d_row.name.strftime('%Y-%m-%d'),
                'Strategy': mode,
                'Ticker': s['ticker'],
                'Side': s['side'],
                'Profit': round(pnl, 2),
                'Z-Score': round(s['score'], 2)
            })

    # 4. EXPORT
    with open(SIGNAL_FILE, 'w') as f:
        json.dump(final_signals, f, indent=4)
        
    if eval_logs:
        df_new = pd.DataFrame(eval_logs)
        df_new.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
        print(f"✅ Uloženo {len(eval_logs)} signálů se Z-score.")

if __name__ == "__main__":
    run_agent()
