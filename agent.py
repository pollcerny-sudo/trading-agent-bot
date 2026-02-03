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
INITIAL_CAPITAL = 10000

def calculate_metrics(equity_curve, daily_pnl_pct):
    if len(equity_curve) < 2: return 0, 0, 0
    equity_ser = pd.Series(equity_curve)
    daily_rets = pd.Series(daily_pnl_pct)
    total_return = (equity_ser.iloc[-1] / equity_ser.iloc[0] - 1) * 100
    std = daily_rets.std()
    sharpe = (daily_rets.mean() / (std + 1e-9)) * np.sqrt(252) if std > 0 else 0
    return total_return, sharpe

def run_trading_agent():
    print(f"🚀 Start agenta: {datetime.now()}")
    
    # 1. STAŽENÍ DAT (Stahujeme 2 roky pro Grid Search + aktuální dny)
    # yfinance vrací data včetně dneška, pokud trh běží
    raw_data = yf.download(TICKERS, period="2y", interval="1d", group_by='ticker')
    
    # Rozdělení na VALIDACI (Grid Search) a OOS (Dnešek)
    # Abychom se vyhnuli Look-ahead biasu, "Dnešek" (poslední řádek) 
    # nesmí být použit pro výpočet signálů, které se mají dnes exekuovat.
    
    processed_data = {}
    for t in TICKERS:
        try:
            d = raw_data[t].dropna()
            # TVORBA INDIKÁTORŮ - Vše posunuto o 1 den (shift), aby ráno u Openu byla data známá
            d['Prev_High20_Strict'] = d['High'].rolling(window=20).max().shift(1)
            d['Prev_Low20_Strict'] = d['Low'].rolling(window=20).min().shift(1)
            d['Prev_Close'] = d['Close'].shift(1)
            d['Prev_Open'] = d['Open'].shift(1)
            d['Prev_High'] = d['High'].shift(1)
            d['Prev_Low'] = d['Low'].shift(1)
            d['Prev_Range'] = (d['High'] - d['Low']).shift(1)
            d['Prev_AvgRange'] = d['Prev_Range'].rolling(window=20).mean()
            d['Prev_Volume'] = d['Volume'].shift(1)
            d['Prev_V_Avg'] = d['Volume'].rolling(window=20).mean().shift(1)
            processed_data[t] = d.dropna()
        except: continue

    # 2. GRID SEARCH
    print("🔎 Optimalizuji parametry...")
    best_params = {m: {'sharpe': -np.inf, 'sl_f': 0.5} for m in ['A', 'B', 'V']}
    
    # Získáme seznam všech unikátních obchodních dnů napříč všemi tickery
    all_dates = pd.DatetimeIndex([])
    for t in processed_data:
        all_dates = all_dates.union(processed_data[t].index)
    
    # Validace na historii (vše kromě posledních 60 dnů)
    valid_days = all_dates[all_dates < all_dates[-60]]
    
    for sl_f in [0.4, 0.6]:
        for mode in ['A', 'B', 'V']:
            cap, eq, rets = INITIAL_CAPITAL, [INITIAL_CAPITAL], []
            for day in valid_days:
                pnl_day = 0
                candidates = []
                for t in TICKERS:
                    # KLÍČOVÁ OPRAVA: Kontrola, zda ticker má data pro tento konkrétní den
                    if t in processed_data and day in processed_data[t].index:
                        row = processed_data[t].loc[day]
                        
                        if mode == 'A':
                            dist_h = abs(row['Prev_Close'] - row['Prev_High20_Strict']) / (row['Prev_AvgRange'] + 1e-9)
                            if dist_h < 0.4: 
                                candidates.append({'t': t, 's': 'Long', 'scr': 1/(dist_h+0.01)})
                        elif mode == 'B':
                            if row['Prev_Volume'] > row['Prev_V_Avg'] * 1.3:
                                side = 'Long' if row['Prev_Close'] > row['Prev_Open'] else 'Short'
                                candidates.append({'t': t, 's': side, 'scr': row['Prev_Volume']})
                        elif mode == 'V':
                            if row['Prev_High'] > row['Prev_High20_Strict']:
                                candidates.append({'t': t, 's': 'Long', 'scr': row['Prev_Range']})
                
                selected = sorted(candidates, key=lambda x: x['scr'], reverse=True)[:3]
                for trade in selected:
                    d_row = processed_data[trade['t']].loc[day]
                    # Simulace profitu (Open -> Close)
                    change = (d_row['Close'] - d_row['Open']) if trade['s'] == 'Long' else (d_row['Open'] - d_row['Close'])
                    pnl_day += (int(10000/d_row['Open']) * change)
                
                rets.append((pnl_day/cap)*100 if cap > 0 else 0)
                cap += pnl_day
                eq.append(cap)
            
            _, sh = calculate_metrics(eq, rets)
            if sh > best_params[mode]['sharpe']:
                best_params[mode] = {'sharpe': sh, 'sl_f': sl_f}
                
    # 3. GENERACE SIGNÁLŮ PRO DNES (Poslední řádek dat)
    # OŠETŘENÍ LOOK-AHEAD: Signály se generují z "včerejších" indikátorů pro "dnešní" Open
    print("🎯 Generuji signály pro dnešní seanci...")
    current_signals = {}
    for mode in ['A', 'B', 'V']:
        candidates = []
        for t in TICKERS:
            # Poslední řádek obsahuje indikátory vypočítané z uzavřených předchozích dní
            row = processed_data[t].iloc[-1]
            if mode == 'A':
                dist_h = abs(row['Prev_Close'] - row['Prev_High20_Strict']) / (row['Prev_AvgRange'] + 1e-9)
                if dist_h < 0.4: candidates.append({'ticker': t, 'action': 'BUY', 'score': 1/dist_h, 'sl_f': best_params[mode]['sl_f']})
            elif mode == 'B':
                if row['Prev_Volume'] > row['Prev_V_Avg'] * 1.3:
                    candidates.append({'ticker': t, 'action': 'BUY' if row['Prev_Close'] > row['Prev_Open'] else 'SELL', 'score': row['Prev_Volume'], 'sl_f': best_params[mode]['sl_f']})
        
        current_signals[mode] = sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]

    with open(SIGNAL_FILE, 'w') as f:
        json.dump(current_signals, f, indent=4)

    # 4. EVALUACE (Zápis včerejších výsledků do CSV)
    # Abychom viděli reálný profit, zapíšeme výsledek dne, který právě skončil (iloc[-1])
    # To se provede jen pokud skript běží večer po Close
    if datetime.now().hour >= 21:
        eval_logs = []
        for mode, trades in current_signals.items():
            for s in trades:
                d_row = processed_data[s['ticker']].iloc[-1]
                # Reálný rozdíl mezi dnešním Open a Close
                pnl = (int(10000/d_row['Open']) * (d_row['Close'] - d_row['Open'] if s['action'] == 'BUY' else d_row['Open'] - d_row['Close'])) - 1.0
                eval_logs.append({
                    'Date': processed_data[s['ticker']].index[-1].strftime('%Y-%m-%d'),
                    'Strategy': mode,
                    'Ticker': s['ticker'],
                    'Side': 'Long' if s['action'] == 'BUY' else 'Short',
                    'Profit': round(pnl, 2)
                })
        
        if eval_logs:
            df_new = pd.DataFrame(eval_logs)
            df_new.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
            print("✅ Výsledky zapsány do CSV.")

if __name__ == "__main__":
    run_trading_agent()
