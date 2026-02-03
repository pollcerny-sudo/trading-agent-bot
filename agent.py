import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime

# --- KONFIGURACE ---
TICKERS = ["AAPL", "^GSPC", "GOOGL", "V", "WMT", "BRK-B", "PLTR", "NVDA", "SPY", "ABBV",
    "BAC", "AMZN", "NFLX", "XOM", "GE", "JPM", "JNJ", "MA", "HD", "AVGO",
    "TSLA", "PG", "CVX", "MSFT", "KO", "META", "ORCL", "ASML", "LLY", "BABA",
    "SAP", "COST", "AMD", "TMUS", "CSCO", "PM", "MS", "QQQ", "AZN", "NVS",
    "UNH", "TM", "IBM", "LIN", "SHEL", "GS", "RTX", "AXP", "CRM", "HSBC",
    "SHOP", "MRK", "RY", "WFC", "PEP", "MCD", "HDB", "UBER", "ABT", "DIS",
    "BX", "VZ", "T", "TMO", "NVO", "BLK", "CYATY", "CAT", "SCHW", "BKNG",
    "INTU", "C", "GEV", "DTEGY", "FMX", "ANET", "BA", "NOW", "TXN", "MUFG",
    "AMGN", "SONY", "SYK", "TJX", "SPGI", "ISRG", "GILD", "AMAT", "QCOM", "ACN",
    "ARM", "APP", "BSX", "PDD", "ADBE", "UL", "SPOT", "DHR", "MU", "COF",
    "HON", "HTHIY", "TTE", "RTNTF", "NEE", "KKR", "BHP", "SAN", "BTI", "DE",
    "PGR", "ETN", "BUD", "PFE", "COP", "ADP", "LOW", "KLAC", "UBS", "TD",
    "MELI", "UNP", "LRCX", "APH", "MDT", "SNY", "CMCSA", "SNPS", "IBN", "MO",
    "ADI", "PANW", "DASH", "MSTR", "SMFG", "IBKR", "CB", "NKE", "BBVA", "CRWD",
    "SO", "WELL", "CEG", "CME", "BAM", "ENB", "ICE", "BN", "RIO", "SE",
    "HOOD", "CFRUY", "MMC", "SBUX", "VRTX", "TT", "DUK", "PH", "PLD", "LMT",
    "INTC", "AMT", "BP", "CTAS", "RBLX", "DELL", "BMY", "ORLY", "CDNS", "MCO",
    "RELX", "RCL", "WM", "HCA", "MGCLY", "SHW", "APO", "MMM", "GD", "TDG",
    "NOC", "MCK", "NTES", "BMO", "CVS", "MFG", "MDLZ", "GLD", "RACE", "AON",
    "MSI", "CI", "AJG", "PBR", "GSK", "COIN", "SCCO", "ECL", "TRI", "UPS",
    "BK", "RSG", "PNC", "FI", "EQIX", "ABNB", "ITW", "CRH", "CRWV", "BAESY",
    "NEM", "EMR", "NGG", "HWM", "JCI", "ITUB", "ING", "BCS", "MAR", "USB",
    "BNS", "VST", "WMB", "CSX", "NET", "CL", "AZO", "ZTS", "CP", "DB",
    "BMWKY", "EPD", "AEM", "CM", "INFY", "LYG", "MRVL", "SPG", "GBTC", "NSC",
    "EQNR", "EOG", "HLT", "MNST", "DEO", "ARES", "ELV", "PYPL", "APD", "CNQ",
    "IAU", "SNOW", "TEL", "AXON", "CNI", "FCX", "ADSK", "AEP", "FTNT", "NU",
    "ATEYY", "URI", "TRV", "PWR", "ET", "NWG", "KMI", "WDAY", "IFNNY", "AMX",
    "CMG", "REGN", "DLR", "ALNY", "NDAQ", "GLW", "AFL", "FAST", "CMI", "CARR",
    "ROP", "BDX", "NXPI", "TFC", "SRE", "PCAR", "VRT", "TRP", "COR", "FDX",
    "MFC", "O", "MET", "PTCAY", "D", "ALL", "IDXX", "E", "GM", "CPNG",
    "FLUT", "MPLX", "PSA", "SLB", "KR", "PAYX", "LHX", "MPC", "LNG", "WCN",
    "AMP", "ROST", "CTVA", "SU", "PSX", "XYZ", "DHI", "JD", "TGT", "TAK"]
LOG_FILE = 'final_backtest_results.csv'
SIGNAL_FILE = 'ibkr_signals.json'

def calculate_metrics(eq, rets):
    if len(eq) < 2: return 0, 0
    eq_ser = pd.Series(eq)
    daily_rets = pd.Series(rets)
    std = daily_rets.std()
    sharpe = (daily_rets.mean() / (std + 1e-9)) * np.sqrt(252)
    return (eq_ser.iloc[-1] / eq_ser.iloc[0] - 1) * 100, sharpe

def run_agent():
    print(f"🚀 Start komplexního agenta: {datetime.now()}")
    
    # 1. STAŽENÍ DAT
    try:
        raw = yf.download(TICKERS, period="2y", interval="1d", group_by='ticker', progress=False)
    except Exception as e:
        print(f"❌ Chyba stahování: {e}"); return
    
    ticker_data = {}
    for t in TICKERS:
        try:
            if t not in raw or raw[t].dropna().empty: continue
            d = raw[t].dropna().copy()
            d['Prev_High20_Strict'] = d['High'].rolling(window=20).max().shift(2)
            d['Prev_Low20_Strict'] = d['Low'].rolling(window=20).min().shift(2)
            d['Prev_Range'] = (d['High'] - d['Low']).shift(1)
            d['Prev_AvgRange'] = d['Prev_Range'].rolling(window=20).mean()
            d['Prev_Close'] = d['Close'].shift(1)
            d['Prev_Open'] = d['Open'].shift(1)
            d['Prev_Volume'] = d['Volume'].shift(1)
            d['Prev_V_Avg'] = d['Volume'].rolling(window=20).mean().shift(1)
            ticker_data[t] = d.dropna()
        except: continue

    # 2. GRID SEARCH PRO STOP-LOSS (Globální optimalizace na In-Sample datech)
    print("🔎 Optimalizuji Stop-Loss parametry...")
    best_sl = {m: 0.5 for m in ['A', 'B', 'V']}
    all_dates = pd.DatetimeIndex([])
    for t in ticker_data: all_dates = all_dates.union(ticker_data[t].index)
    valid_days = all_dates[all_dates < all_dates[-60]] # Historie pro test

    for mode in ['A', 'B', 'V']:
        best_sharpe = -np.inf
        for sl_f in [0.3, 0.5, 0.8]:
            cap, eq, rets = 10000, [10000], []
            for day in valid_days:
                pnl_day = 0
                for t in ticker_data:
                    if day not in ticker_data[t].index: continue
                    row = ticker_data[t].loc[day]
                    
                    # Signal check (zjednodušený pro Grid Search)
                    hit = False
                    side = 'Long'
                    if mode == 'A' and abs(row['Prev_Close'] - row['Prev_High20_Strict']) / row['Prev_AvgRange'] < 0.4: hit = True
                    if mode == 'B' and row['Prev_Volume'] > row['Prev_V_Avg'] * 1.5: hit = True
                    if mode == 'V' and row['High'] > row['Prev_High20_Strict']: hit = True
                    
                    if hit:
                        ent = row['Open']
                        sl_dist = row['Prev_Range'] * sl_f
                        # Simulace SL přes High/Low dne
                        if row['Low'] <= ent - sl_dist: ext = ent - sl_dist
                        else: ext = row['Close']
                        pnl_day += (int(2000/ent) * (ext - ent)) # Fixní sázka 2k na pozici
                
                rets.append((pnl_day/cap)*100); cap += pnl_day; eq.append(cap)
            
            _, sh = calculate_metrics(eq, rets)
            if sh > best_sharpe:
                best_sharpe = sh
                best_sl[mode] = sl_f

    # 3. Z-SCORE SKÓROVÁNÍ TICKERŮ (S využitím optimálního SL)
    print(f"📊 Počítám Z-score s optimalizovanými SL: {best_sl}")
    ticker_performance = {m: {} for m in ['A', 'B', 'V']}
    for t, df in ticker_data.items():
        hist_df = df.iloc[:-5]
        for mode in ['A', 'B', 'V']:
            # Najdeme historické dny se signálem
            if mode == 'A': sigs = hist_df[abs(hist_df['Prev_Close'] - hist_df['Prev_High20_Strict']) / hist_df['Prev_AvgRange'] < 0.4]
            elif mode == 'B': sigs = hist_df[hist_df['Prev_Volume'] > hist_df['Prev_V_Avg'] * 1.5]
            else: sigs = hist_df[hist_df['High'].shift(1) > hist_df['Prev_High20_Strict']]
            
            if not sigs.empty:
                # Simulujeme výnosy s optimálním SL
                sl_dist = sigs['Prev_Range'] * best_sl[mode]
                pnl_pct = (sigs['Close'] - sigs['Open']) / sigs['Open'] # Zjednodušeně EOD
                ticker_performance[mode][t] = pnl_pct.mean() / (pnl_pct.std() + 1e-9) if len(pnl_pct) > 3 else 0

    # 4. GENERACE SIGNÁLŮ PRO DNES
    final_signals, eval_logs = {}, []
    for mode in ['A', 'B', 'V']:
        candidates = []
        for t, df in ticker_data.items():
            row = df.iloc[-1]
            is_sig = False
            if mode == 'A' and abs(row['Prev_Close'] - row['Prev_High20_Strict']) / row['Prev_AvgRange'] < 0.4: is_sig = True
            if mode == 'B' and row['Prev_Volume'] > row['Prev_V_Avg'] * 1.5: is_sig = True
            if mode == 'V' and row['High'] > row['Prev_High20_Strict']: is_sig = True
            
            if is_sig:
                candidates.append({'ticker': t, 'side': 'Long', 'score': ticker_performance[mode].get(t, 0)})
        
        sel = sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
        final_signals[mode] = sel
        for s in sel:
            d_row = ticker_data[s['ticker']].iloc[-1]
            pnl = (int(10000/d_row['Open']) * (d_row['Close'] - d_row['Open'])) - 1.0
            eval_logs.append({'Date': d_row.name.strftime('%Y-%m-%d'), 'Strategy': mode, 'Ticker': s['ticker'], 'Side': s['side'], 'Profit': round(pnl, 2), 'Z-Score': round(s['score'], 2), 'SL_Factor': best_sl[mode]})

    # 5. ULOŽENÍ
    with open(SIGNAL_FILE, 'w') as f: json.dump(final_signals, f, indent=4)
    if eval_logs: pd.DataFrame(eval_logs).to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
    print("✅ Hotovo.")

if __name__ == "__main__":
    run_agent()
