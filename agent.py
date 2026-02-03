import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime

TICKERS = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "GOOGL", "AMZN", "META", "PLTR", "SPY", "QQQ", "NFLX", "AVGO", "SMCI"]
LOG_FILE = 'final_backtest_results.csv'
SIGNAL_FILE = 'ibkr_signals.json'
COMMISSION_PCT = 0.001  # 0.1% commission per trade (buy + sell = 0.2% total)

def calculate_z_score(profits):
    if len(profits) < 3: return 0
    mean = np.mean(profits)
    std = np.std(profits, ddof=1)  # Sample standard deviation
    return mean / (std + 1e-9)

def run_agent():
    print(f"🚀 Spouštím agenta s historickým Z-score skórováním...")
    
    # 1. Stažení dat (potřebujeme delší historii pro Z-score)
    print(f"📥 Stahuji data pro {len(TICKERS)} tickerů...")
    try:
        raw = yf.download(TICKERS, period="2y", interval="1d", group_by='ticker', progress=False)
    except Exception as e:
        print(f"❌ Chyba při stahování dat: {e}")
        return
    
    ticker_data = {}
    failed_tickers = []
    
    for t in TICKERS:
        try:
            # Kontrola, zda ticker má data
            if t not in raw.columns.get_level_values(0):
                failed_tickers.append(t)
                print(f"⚠️  Ticker {t}: Žádná data nenalezena")
                continue
                
            d = raw[t].dropna().copy()
            
            # Kontrola minimálního množství dat
            if len(d) < 30:
                failed_tickers.append(t)
                print(f"⚠️  Ticker {t}: Nedostatek dat ({len(d)} dní)")
                continue
            
            # Přidání technických indikátorů
            d['Prev_High20_Strict'] = d['High'].rolling(window=20).max().shift(2)
            d['Prev_Low20_Strict'] = d['Low'].rolling(window=20).min().shift(2)
            d['Prev_Range'] = (d['High'] - d['Low']).shift(1)
            d['Prev_AvgRange'] = d['Prev_Range'].rolling(window=20).mean()
            d['Prev_Close'] = d['Close'].shift(1)
            d['Prev_Open'] = d['Open'].shift(1)
            d['Prev_Volume'] = d['Volume'].shift(1)
            d['Prev_V_Avg'] = d['Volume'].rolling(window=20).mean().shift(1)
            d['Prev_High'] = d['High'].shift(1)  # Přidáno pro konzistenci
            
            # Denní výnos pro backtest (Open-to-Close)
            d['Day_Return_Pct'] = (d['Close'] - d['Open']) / d['Open']
            
            # Odstranění NaN hodnot
            d_clean = d.dropna()
            
            if len(d_clean) < 30:
                failed_tickers.append(t)
                print(f"⚠️  Ticker {t}: Nedostatek validních dat po výpočtech ({len(d_clean)} dní)")
                continue
                
            ticker_data[t] = d_clean
            print(f"✅ Ticker {t}: Načteno {len(d_clean)} dní dat")
            
        except Exception as e:
            failed_tickers.append(t)
            print(f"❌ Ticker {t}: Chyba při zpracování - {str(e)}")
            continue
    
    if not ticker_data:
        print("❌ Žádná validní data pro analýzu. Ukončuji.")
        return
    
    print(f"\n📊 Úspěšně načteno: {len(ticker_data)} tickerů")
    if failed_tickers:
        print(f"⚠️  Selhalo: {len(failed_tickers)} tickerů: {', '.join(failed_tickers)}")

    # 2. HISTORICKÉ SKÓROVÁNÍ TICKERŮ (Z-SCORE)
    # Zjistíme, jak který ticker historicky fungoval, když nastal signál
    print(f"\n🧮 Počítám historické Z-score pro každou strategii...")
    ticker_performance = {m: {} for m in ['A', 'B', 'V']}
    
    for t in ticker_data:
        df = ticker_data[t]
        # Vynecháme posledních 5 dní, abychom neměli bias
        hist_df = df.iloc[:-5] 
        
        # --- Strategie A (Mean Reversion / Setup) ---
        sig_a = hist_df[abs(hist_df['Prev_Close'] - hist_df['Prev_High20_Strict']) / hist_df['Prev_AvgRange'] < 0.4]
        ticker_performance['A'][t] = calculate_z_score(sig_a['Day_Return_Pct'])

        # --- Strategie B (Volume Breakout) ---
        sig_b = hist_df[hist_df['Prev_Volume'] > hist_df['Prev_V_Avg'] * 1.5]
        ticker_performance['B'][t] = calculate_z_score(sig_b['Day_Return_Pct'])

        # --- Strategie V (Trend Breakout) ---
        # OPRAVENO: Používáme Prev_High místo shift inline
        sig_v = hist_df[hist_df['Prev_High'] > hist_df['Prev_High20_Strict']]
        ticker_performance['V'][t] = calculate_z_score(sig_v['Day_Return_Pct'])

    # 3. GENERACE SIGNÁLŮ PRO DNES
    print(f"\n🎯 Generuji signály pro dnešní obchodování...")
    final_signals = {}
    eval_logs = []
    
    for mode in ['A', 'B', 'V']:
        candidates = []
        for t in ticker_data:
            row = ticker_data[t].iloc[-1]
            # Podmínka pro dnešní signál
            is_signal = False
            side = 'Long'
            
            if mode == 'A':
                dist_h = abs(row['Prev_Close'] - row['Prev_High20_Strict']) / (row['Prev_AvgRange'] + 1e-9)
                if dist_h < 0.4: 
                    is_signal = True
                    side = 'Long'
            elif mode == 'B':
                if row['Prev_Volume'] > row['Prev_V_Avg'] * 1.5:
                    is_signal = True
                    side = 'Long' if row['Prev_Close'] > row['Prev_Open'] else 'Short'
            elif mode == 'V':
                # OPRAVENO: row je již Series, ne DataFrame - bez .iloc[-1]
                if row['Prev_High'] > row['Prev_High20_Strict']:
                    is_signal = True
                    side = 'Long'

            if is_signal:
                # Klíčová část: Použijeme historické Z-score pro tento ticker v této strategii
                z_score = ticker_performance[mode].get(t, 0)
                candidates.append({
                    'ticker': t, 
                    'side': side, 
                    'score': z_score, # Řadíme podle historické úspěšnosti
                    'current_vol': row['Prev_Volume'] / row['Prev_V_Avg']
                })
        
        # Výběr TOP 3 podle historického Z-score
        sel = sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
        final_signals[mode] = sel
        
        print(f"  Strategie {mode}: {len(candidates)} kandidátů → vybráno TOP {len(sel)}")
        
        # Příprava logu pro CSV
        for s in sel:
            d_row = ticker_data[s['ticker']].iloc[-1]
            
            # OPRAVENO: Realistický výpočet komisí (0.1% na nákup + 0.1% na prodej)
            shares = int(10000 / d_row['Open'])
            gross_pnl = shares * (d_row['Close'] - d_row['Open'] if s['side'] == 'Long' else d_row['Open'] - d_row['Close'])
            commission = 10000 * COMMISSION_PCT * 2  # Buy + Sell
            net_pnl = gross_pnl - commission
            
            eval_logs.append({
                'Date': d_row.name.strftime('%Y-%m-%d'),
                'Strategy': mode,
                'Ticker': s['ticker'],
                'Side': s['side'],
                'Type': 'EOD',
                'Profit': round(net_pnl, 2),
                'Z-Score': round(s['score'], 2)
            })

    # Uložení
    try:
        with open(SIGNAL_FILE, 'w') as f:
            json.dump(final_signals, f, indent=4)
        print(f"\n💾 Signály uloženy do: {SIGNAL_FILE}")
    except Exception as e:
        print(f"❌ Chyba při ukládání signálů: {e}")
        
    if eval_logs:
        try:
            df_new = pd.DataFrame(eval_logs)
            df_new.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
            print(f"✅ Vybráno {len(eval_logs)} obchodů na základě historického Z-score.")
            print(f"📝 Log uložen do: {LOG_FILE}")
        except Exception as e:
            print(f"❌ Chyba při ukládání logu: {e}")
    else:
        print("⚠️  Žádné signály k zalogování.")

if __name__ == "__main__":
    run_agent()
        
