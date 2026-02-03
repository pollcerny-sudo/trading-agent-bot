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

def run_agent():
    print(f"Stahuji data pro: {TICKERS}")
    # Stahujeme data (group_by='ticker' zajistí stabilitu struktury)
    df_daily = yf.download(TICKERS, period="40d", interval="1d", group_by='ticker')
    
    signals = {"Strategy_A": [], "Strategy_B": [], "Strategy_V": []}
    
    for t in TICKERS:
        try:
            d = df_daily[t].dropna()
            if len(d) < 21: continue
            
            # Data k včerejšímu uzavření (poslední řádek je index -1)
            high20 = d['High'].rolling(window=20).max().iloc[-2]
            low20 = d['Low'].rolling(window=20).min().iloc[-2]
            prev_close = d['Close'].iloc[-2]
            prev_open = d['Open'].iloc[-2]
            prev_high = d['High'].iloc[-2]
            prev_low = d['Low'].iloc[-2]
            prev_range = d['High'].iloc[-2] - d['Low'].iloc[-2]
            avg_range = (d['High'] - d['Low']).rolling(window=20).mean().iloc[-2]
            prev_vol = d['Volume'].iloc[-2]
            avg_vol = d['Volume'].rolling(window=20).mean().iloc[-2]

            # --- STRATEGIE A (Setup) ---
            dist_h = abs(prev_close - high20) / (avg_range + 1e-9)
            dist_l = abs(prev_close - low20) / (avg_range + 1e-9)
            if dist_h < 0.4: signals["Strategy_A"].append({'ticker': t, 'side': 'Long', 'score': 1/(dist_h+0.01), 'range': prev_range, 'price': prev_close})
            elif dist_l < 0.4: signals["Strategy_A"].append({'ticker': t, 'side': 'Short', 'score': 1/(dist_l+0.01), 'range': prev_range, 'price': prev_close})

            # --- STRATEGIE B (Volume) ---
            if prev_vol > avg_vol * 1.3:
                side = 'Long' if prev_close > prev_open else 'Short'
                signals["Strategy_B"].append({'ticker': t, 'side': side, 'score': prev_vol, 'range': prev_range, 'price': prev_close})

            # --- STRATEGIE V (Reactive) ---
            if prev_high > high20:
                signals["Strategy_V"].append({'ticker': t, 'side': 'Long', 'score': prev_range, 'range': prev_range, 'price': prev_close})
            elif prev_low < low20:
                signals["Strategy_V"].append({'ticker': t, 'side': 'Short', 'score': prev_range, 'range': prev_range, 'price': prev_close})
        except: continue

    # Formátování pro JSON
    output_signals = {}
    for mode in signals:
        top3 = sorted(signals[mode], key=lambda x: x['score'], reverse=True)[:3]
        formatted = []
        for s in top3:
            # Předběžný Stop Loss pro informaci
            sl_est = s['price'] - (s['range']*0.5) if s['side']=='Long' else s['price'] + (s['range']*0.5)
            formatted.append({
                'ticker': s['ticker'],
                'action': 'BUY' if s['side'] == 'Long' else 'SELL',
                'allocation_usd': 10000,
                'est_sl': round(sl_est, 2),
                'sl_factor': 0.5
            })
        output_signals[mode] = formatted

    with open(SIGNAL_FILE, 'w') as f:
        json.dump(output_signals, f, indent=4)
    print("✅ Signály vygenerovány.")

if __name__ == "__main__":
    run_agent()
