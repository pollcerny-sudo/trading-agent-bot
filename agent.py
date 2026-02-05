import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
import sys

# === MANUAL TICKER GROUPING ===
BIG_TICKERS = ["AAPL", "^GSPC", "GOOGL", "V", "WMT", "BRK-B", "PLTR", "NVDA", "SPY", "ABBV",    "BAC", "AMZN", "NFLX", "XOM", "GE", "JPM", "JNJ", "MA", "HD", "AVGO",
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
SMALL_TICKERS = ["IP", "DSFIY", "CQP", "WSM", "TDY", "FTS", "HPQ", "FE", "TOST",    "INSM", "FIX", "SHG", "EXPE", "DG", "CNP", "VRSN", "JBL", "EQR", "TYL",
    "AFRM", "PPG", "TU", "YAHOY", "ES", "PTC", "CLS", "FCNCA", "LI", "PHM",
    "STE", "PINS", "MKL", "DOV", "NTRS", "TPG", "DRI", "STM", "HBAN", "DLTR",
    "FOXA", "CG", "CINF", "SBAC", "NJDCY", "RSHGY", "PUBGY", "TROW", "EXE", "LULU",
    "ULTA", "LDOS", "SW", "BCE", "TPR", "TELNY", "HUBB", "FUTU", "KGC", "PHYS",
    "CDW", "CHD", "RF", "LH", "CPAY", "HUBS", "EIX", "PMDIY", "NVR", "CMS",
    "RBA", "AMCR", "NTAP", "GIB", "CKHGF", "NTRA", "PODD", "CYBR", "NOK", "SSNC",
    "AS", "DKNG", "NMR", "ASX", "PBA", "BEKE", "ZM", "LII", "CRDO", "DVN",
    "RPRX", "HKHHY", "RKLB", "TRMB", "AER", "TSN", "CFG", "ESLT", "ON", "DGX",
    "GRAB", "VIV", "FLEX", "TPL", "BAP", "ZBH", "FSLR", "L", "CHKP", "NI",
    "KEY", "GDDY", "GEN", "GPN", "TLK", "ZG", "TS", "ERIE", "CW", "CASY",
    "EBR", "RCI", "PSTG", "TEVA", "BIIB", "STLD", "XPEV", "SMMT", "FFIV", "GPC",
    "KSPI", "IOT", "GFL", "MKC", "EDPFY", "NTNX", "INVH", "RL", "DKS", "USFD",
    "COHR", "KOF", "IHG", "OMVKY", "KEP", "CRBG", "PKG", "CTRA", "GFS", "WY",
    "ESS", "EC", "NBIS", "BURL", "J", "HAL", "GWRE", "TER", "ASTS", "NWSA",
    "RBRK", "PNR", "PFG", "TRU", "EVRG", "TLN", "IT", "PKX", "WST", "LNT",
    "MAA", "WSO", "UMC", "ZBRA", "JHX", "WMG", "SNA", "MDB", "YUMC", "SGI",
    "BG", "EQH", "NPXYY", "ARNNY", "SN", "WAT", "IFF", "FNF", "SUI", "BWXT",
    "EXPD", "BEP", "BNT", "INCY", "CSL", "FTV", "TECK", "ZTO", "ONON", "ACM",
    "LYB", "BSY", "LUV", "VDMCY", "SNN", "ARCC", "RYAN", "HRL", "DOW", "RPM",
    "PFGC", "APTV", "CNH", "U", "HOLX", "SBS", "JOBY", "BLDR", "RS", "OKTA",
    "BCH", "ILMN", "CLX", "NVT", "DPZ", "THC", "XPO", "TKO", "WWD", "CHWY",
    "SOAGY", "APG", "BBY", "AMH", "DECK", "LOGI", "TWLO", "MAS", "FTI", "FTAI",
    "EMA", "BALL", "BF-A", "RIVN", "COO", "WPC", "DUOL", "KIM", "HNGKY", "MTZ",
    "OMC", "WES", "UDR", "ALLE", "GGG", "BIP", "SFM", "EWBC", "BJ", "DT",
    "GMAB", "JBHT", "HLI", "FMS", "CHRW", "DOCU", "EG", "AVY", "CELH", "TXT",
    "FDS", "CIEN", "CF", "UTHR", "ULS", "REG", "CART", "NLY", "WF", "LTM",
    "BEN", "BAH", "LECO", "YPF", "QXO", "FNMA", "ITT", "JLL", "H", "RBC",
    "MEDP", "CLH", "CNM", "GLPI", "SUZ", "CNA", "ICLR", "MP", "NBIX", "RTO",
    "PNDRY", "IONQ", "CX", "PAC", "CRS", "MANH", "RKUNY", "SOLV", "JEF", "FN",
    "CNC", "AVAV", "STN", "OC", "MGA", "PAA", "ARE", "BAX", "TOL", "SNX",
    "SNAP", "ROKU", "OHI", "NDSN", "CIB", "ELS", "COOP", "IEX", "PAG", "EHC",
    "UNM", "BLD", "RGA", "SJM", "YMM", "EVR", "PAYC", "LAMR", "ASND", "PPC",
    "GTM", "CCK", "WYNN", "SQM", "DOC", "ALLY", "SLV", "AEG", "TXRH", "POOL",
    "JKHY", "SF", "VTRS", "DOCS", "CR", "BSAC", "DLAKY", "AUR", "PAAS", "KTOS",
    "GNRC", "RNR", "SCI", "RGLD", "ENTG", "OKLO", "BROS", "GL", "RDY", "MBLY",
    "SOTGY", "CPT", "TEM", "FHN", "UHS", "BXP", "DRS", "PNW", "SWK", "CHYM",
    "VNOM", "DSEEY", "HAS", "NCLH", "HST", "MORN", "HIMS", "FRHC", "LKNCY", "ERJ",
    "AGI", "SAIL", "PR", "SEIC", "CRRFY", "ACI", "BMRN", "WMS", "PUGBY", "GLXY",
    "UUGRY", "UWMC", "QGEN", "AFG", "SWKS", "RDEIY", "WTRG", "DTM", "AIZ", "WBA",
    "ATI", "HII", "NCLTY", "CACI", "AIT", "BZLFY", "EXEL", "ARMK", "WLK", "OVV",
    "RVTY", "WCC", "MLI", "KNSL", "ALGN", "LINE", "HTHT", "AKAM", "AOS", "TAP",
    "GME", "UHAL", "AGNC", "BPYPP", "HMY", "BZ", "NIO", "APPF", "TIMB", "KT",
    "COKE", "CSXXY", "AYI", "ASR", "MOS", "SFD", "AR", "WBS", "DOX", "MRNA",
    "MGM", "SARO", "KAIKY", "CAVA", "TTEK", "PEN", "BE", "CPB", "W", "SSB",
    "SKX", "BILI", "PCOR", "NYT", "IVZ", "ORI", "PCTY", "RRX", "WING", "ENSG",
    "CAI", "XP", "MTSI", "BBIO", "MYTHY", "STRL", "PSO", "CAE", "FYBR", "DVA",
    "ATR", "SPXC", "IPG", "ALB", "PRMB", "OGE", "CAG", "NICE", "WTS", "WAL",
    "MMYT", "AES", "KVYO", "REXR", "BIRK", "VIRT", "CUBE", "PLNT", "VRNA", "GTLS",
    "MNDY"]
ETF_TICKERS = ["SPY", "QQQ"]
ALL_TICKERS = BIG_TICKERS + SMALL_TICKERS + ETF_TICKERS

STRATEGY_TICKER_GROUPS = {
    'A': BIG_TICKERS + ETF_TICKERS,
    'B': BIG_TICKERS + ETF_TICKERS,
    'V': SMALL_TICKERS,
    'M': SMALL_TICKERS
}

# Output directory
OUTPUT_DIR = os.getcwd()
LOG_FILE = os.path.join(OUTPUT_DIR, 'final_backtest_results.csv')
SIGNAL_FILE = os.path.join(OUTPUT_DIR, 'ibkr_signals.json')
OPTIMIZATION_FILE = os.path.join(OUTPUT_DIR, 'sl_optimization_results.json')
BACKTEST_FILE = os.path.join(OUTPUT_DIR, 'backtest_60d_results.json')
COMMISSION_PCT = 0.0001

SL_GRID = [0.4, 0.6, 0.8, 1.6, 3.2]
ALLOCATION_USD = 10000
BACKTEST_DAYS = 60

def calculate_z_score(profits):
    if len(profits) < 3: return 0
    mean = np.mean(profits)
    std = np.std(profits, ddof=1)
    return mean / (std + 1e-9)

def simulate_trade_with_sl(row, side, sl_factor, commission_pct=0.001):
    """Simuluje obchod s stop lossem"""
    entry_price = row['Open']
    exit_price = row['Close']
    sl_distance = sl_factor * row['Prev_AvgRange']
    shares = int(ALLOCATION_USD / entry_price)
    
    if side == 'Long':
        sl_price = entry_price - sl_distance
        if row['Low'] <= sl_price:
            exit_price = sl_price
            hit_sl = True
        else:
            hit_sl = False
        gross_pnl = shares * (exit_price - entry_price)
    else:  # Short
        sl_price = entry_price + sl_distance
        if row['High'] >= sl_price:
            exit_price = sl_price
            hit_sl = True
        else:
            hit_sl = False
        gross_pnl = shares * (entry_price - exit_price)
    
    commission = ALLOCATION_USD * commission_pct * 2
    net_pnl = gross_pnl - commission
    
    return net_pnl, hit_sl

def evaluate_todays_signals_on_5min_data():
    """
    EVENING MODE: Načte dnešní signály z JSON a vyhodnotí je proti 5min datům.
    Vrací updated signály s výsledky a profitem.
    """
    print(f"\n{'='*70}")
    print(f"📊 EVENING MODE: VYHODNOCENÍ DNEŠNÍCH SIGNÁLŮ")
    print(f"{'='*70}\n")
    
    # 1. Načti dnešní signály
    if not os.path.exists(SIGNAL_FILE):
        print(f"⚠️  Žádné signály k vyhodnocení (soubor {SIGNAL_FILE} neexistuje)")
        return []
    
    try:
        with open(SIGNAL_FILE, 'r') as f:
            todays_signals = json.load(f)
        print(f"✅ Načteny signály z {SIGNAL_FILE}")
    except Exception as e:
        print(f"❌ Chyba při načítání signálů: {e}")
        return []
    
    # 2. Stáhni 5min data pro dnes
    print(f"\n📥 Stahuji 5min data pro dnešní evaluaci...")
    
    all_tickers_in_signals = set()
    for strategy, signals in todays_signals.items():
        for sig in signals:
            all_tickers_in_signals.add(sig['ticker'])
    
    if not all_tickers_in_signals:
        print(f"⚠️  Žádné tickery k vyhodnocení")
        return []
    
    print(f"   Tickery: {', '.join(all_tickers_in_signals)}")
    
    # Stáhni 5min data pro dnes (+ včera pro případ že dnes ještě není complete)
    ticker_5min_data = {}
    for ticker in all_tickers_in_signals:
        try:
            # Stáhni 5 dní 5min dat (pokryje dnešek i včerejšek)
            data = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
            if not data.empty:
                ticker_5min_data[ticker] = data
                print(f"   ✅ {ticker}: {len(data)} 5min barov")
            else:
                print(f"   ⚠️  {ticker}: Žádná 5min data")
        except Exception as e:
            print(f"   ❌ {ticker}: Chyba - {e}")
    
    # 3. Vyhodnoť každý signál
    print(f"\n📊 Vyhodnocuji signály...")
    evaluated_trades = []
    
    for strategy, signals in todays_signals.items():
        if not signals:
            continue
            
        print(f"\n🎯 Strategie {strategy}:")
        
        for sig in signals:
            ticker = sig['ticker']
            action = sig['action']
            sl_factor = sig['sl_factor']
            
            if ticker not in ticker_5min_data:
                print(f"   ⚠️  {ticker}: Žádná 5min data k vyhodnocení")
                continue
            
            df_5min = ticker_5min_data[ticker]
            
            # Dnešní data (poslední trading day)
            today = df_5min.index[-1].date()
            todays_bars = df_5min[df_5min.index.date == today]
            
            if todays_bars.empty:
                print(f"   ⚠️  {ticker}: Žádné dnešní 5min bary")
                continue
            
            # Entry = první bar (Open cena)
            entry_bar = todays_bars.iloc[0]
            entry_price = float(entry_bar['Open'].iloc[0] if hasattr(entry_bar['Open'], 'iloc') else entry_bar['Open'])
            
            # Exit = poslední bar (Close cena) nebo SL během dne
            exit_bar = todays_bars.iloc[-1]
            exit_price = float(exit_bar['Close'].iloc[0] if hasattr(exit_bar['Close'], 'iloc') else exit_bar['Close'])
            shares = int(ALLOCATION_USD / entry_price)
            
            # Simulace SL během dne
            hit_sl = False
            
            # Potřebujeme Prev_AvgRange - použijeme 1D data
            try:
                daily_data = yf.download(ticker, period="30d", interval="1d", progress=False, auto_adjust=True)
                if len(daily_data) >= 20:
                    avg_range = float((daily_data['High'] - daily_data['Low']).rolling(20).mean().iloc[-1])
                    sl_distance = sl_factor * avg_range
                else:
                    # Fallback: 2% range
                    sl_distance = entry_price * 0.02
            except:
                sl_distance = entry_price * 0.02
            
            if action == 'Long':
                sl_price = entry_price - sl_distance
                # Zkontroluj každý 5min bar
                for idx, bar in todays_bars.iterrows():
                    bar_low = float(bar['Low'].iloc[0] if hasattr(bar['Low'], 'iloc') else bar['Low'])
                    if bar_low <= sl_price:
                        exit_price = sl_price
                        hit_sl = True
                        break
                gross_pnl = shares * (exit_price - entry_price)
            else:  # Short
                sl_price = entry_price + sl_distance
                for idx, bar in todays_bars.iterrows():
                    bar_high = float(bar['High'].iloc[0] if hasattr(bar['High'], 'iloc') else bar['High'])
                    if bar_high >= sl_price:
                        exit_price = sl_price
                        hit_sl = True
                        break
                gross_pnl = shares * (entry_price - exit_price)
            
            commission = ALLOCATION_USD * COMMISSION_PCT * 2
            net_pnl = gross_pnl - commission
            
            # Log výsledek
            evaluated_trades.append({
                'Date': today.strftime('%Y-%m-%d'),
                'Strategy': strategy,
                'Ticker': ticker,
                'Side': action,
                'Type': 'INTRADAY-5MIN',
                'Entry': round(entry_price, 2),
                'Exit': round(exit_price, 2),
                'Profit': round(net_pnl, 2),
                'SL-Factor': sl_factor,
                'Hit-SL': hit_sl,
                'Ticker-Group': sig.get('ticker_group', 'N/A'),
                'Num-5min-Bars': len(todays_bars)
            })
            
            status = "🛑 SL Hit" if hit_sl else "✅ Closed"
            print(f"   {ticker:6s} {action:5s}: Entry=${entry_price:.2f} Exit=${exit_price:.2f} "
                  f"P&L=${net_pnl:>+8.2f} {status}")
    
    # 4. Ulož výsledky do CSV
    if evaluated_trades:
        try:
            df_new = pd.DataFrame(evaluated_trades)
            df_new.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
            print(f"\n✅ Vyhodnoceno {len(evaluated_trades)} obchodů (uloženo do {LOG_FILE})")
            
            total_pnl = sum(t['Profit'] for t in evaluated_trades)
            print(f"   💰 Celkový dnešní P&L: ${total_pnl:,.2f}")
        except Exception as e:
            print(f"❌ Chyba při ukládání evaluace: {e}")
    else:
        print(f"\n⚠️  Žádné obchody k vyhodnocení")
    
    return evaluated_trades

def optimize_sl_for_ticker_strategy(df, strategy_mode, ticker):
    """Grid search pro optimální stop loss"""
    results = []
    
    for sl_factor in SL_GRID:
        trades = []
        
        for idx, row in df.iterrows():
            if strategy_mode == 'A':
                side = 'Long'
            elif strategy_mode == 'B':
                side = 'Long' if row['Prev_Close'] > row['Prev_Open'] else 'Short'
            elif strategy_mode == 'V':
                side = 'Long'
            elif strategy_mode == 'M':
                side = 'Long'
            
            profit, hit_sl = simulate_trade_with_sl(row, side, sl_factor, COMMISSION_PCT)
            trades.append({'profit': profit, 'hit_sl': hit_sl})
        
        if not trades:
            continue
        
        profits = [t['profit'] for t in trades]
        results.append({
            'sl_factor': sl_factor,
            'total_profit': sum(profits),
            'win_rate': sum(1 for p in profits if p > 0) / len(profits) * 100,
            'avg_profit': np.mean(profits),
            'sharpe': calculate_z_score(profits),
            'num_trades': len(trades),
            'sl_hit_rate': sum(1 for t in trades if t['hit_sl']) / len(trades) * 100
        })
    
    if not results:
        return 0.5, None
    
    best = max(results, key=lambda x: x['sharpe'])
    return best['sl_factor'], best

def run_backtest_60d(ticker_data, optimized_sl, ticker_performance):

    print(f"\n{'='*70}")
    print(f"📊 BACKTEST POSLEDNÍCH {BACKTEST_DAYS} DNÍ")
    print(f"{'='*70}\n")

    backtest_results = {}

    for mode in ['A','B','V','M']:

        print(f"\n🎯 Strategie {mode}")
        print("-"*70)

        allowed = STRATEGY_TICKER_GROUPS[mode]
        filtered = {t:ticker_data[t] for t in allowed if t in ticker_data}

        equity = 10000
        equity_curve = [equity]
        trades = []

        for day_offset in range(BACKTEST_DAYS,0,-1):

            candidates = []

            for t,df in filtered.items():

                if len(df) < day_offset+1:
                    continue

                prev = df.iloc[-(day_offset+1)]
                now  = df.iloc[-day_offset]

                signal=False
                side="Long"
                score=0

                if mode=="A":
                    d = abs(prev['Prev_Close']-prev['Prev_High20_Strict'])/(prev['Prev_AvgRange']+1e-9)
                    if d<0.4:
                        signal=True
                        score=ticker_performance[mode].get(t,0)

                elif mode=="B":
                    if prev['Prev_Volume']>prev['Prev_V_Avg']*1.5:
                        signal=True
                        side="Long" if prev['Prev_Close']>prev['Prev_Open'] else "Short"
                        score=ticker_performance[mode].get(t,0)

                elif mode=="V":
                    if prev['Prev_High']>prev['Prev_High20_Strict']:
                        signal=True
                        score=ticker_performance[mode].get(t,0)

                elif mode=="M":
                    signal=True
                    score=prev['Day_Return_Pct']

                if signal:
                    sl = optimized_sl[mode].get(t,0.5)
                    candidates.append({
                        "ticker":t,
                        "side":side,
                        "score":score,
                        "sl":sl,
                        "row":now
                    })

            top3 = sorted(candidates,key=lambda x:x["score"],reverse=True)[:3]

            day_pnl=0

            for tr in top3:

                pnl,hit = simulate_trade_with_sl(
                    tr["row"],
                    tr["side"],
                    tr["sl"],
                    COMMISSION_PCT
                )

                day_pnl += pnl

                trades.append({
                    "date":str(tr["row"].name),
                    "ticker":tr["ticker"],
                    "side":tr["side"],
                    "profit":round(pnl,2),
                    "hit_sl":hit
                })

            equity += day_pnl
            equity_curve.append(round(equity,2))

        dd = calculate_max_drawdown(equity_curve)

        backtest_results[mode] = {
            "equity_curve": equity_curve,
            "num_trades": len(trades),
            "total_profit": round(equity-10000,2),
            "max_drawdown": round(dd,2),
            "trades": trades
        }

        print("profit:",round(equity-10000,2),
              "trades:",len(trades),
              "maxDD:",round(dd,2))

    return backtest_results
        

def calculate_max_drawdown(equity_curve):
    if len(equity_curve) < 2:
        return 0
    peak = equity_curve[0]
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd

def generate_signals_for_tomorrow(ticker_data, optimized_sl, ticker_performance):
    """
    Generuje signály pro zítřejší den.
    """
    print(f"\n{'='*70}")
    print(f"🎯 GENERACE SIGNÁLŮ PRO ZÍTŘEK")
    print(f"{'='*70}\n")
    
    final_signals = {}
    
    for mode in ['A', 'B', 'V', 'M']:
        allowed_tickers = STRATEGY_TICKER_GROUPS[mode]
        filtered_data = {t: ticker_data[t] for t in allowed_tickers if t in ticker_data}
        ticker_group = 'BIG + ETF' if mode in ['A', 'B'] else 'SMALL'
        
        print(f"\nStrategie {mode} ({ticker_group}):")
        
        candidates = []
        
        for t in filtered_data:
            row = ticker_data[t].iloc[-1]
            is_signal = False
            side = 'Long'
            score = 0
            
            if mode == 'A':
                dist_h = abs(row['Prev_Close'] - row['Prev_High20_Strict']) / (row['Prev_AvgRange'] + 1e-9)
                if dist_h < 0.4: 
                    is_signal = True
                    side = 'Long'
                    score = ticker_performance[mode].get(t, 0)
            elif mode == 'B':
                if row['Prev_Volume'] > row['Prev_V_Avg'] * 1.5:
                    is_signal = True
                    side = 'Long' if row['Prev_Close'] > row['Prev_Open'] else 'Short'
                    score = ticker_performance[mode].get(t, 0)
            elif mode == 'V':
                if row['Prev_High'] > row['Prev_High20_Strict']:
                    is_signal = True
                    side = 'Long'
                    score = ticker_performance[mode].get(t, 0)
            elif mode == 'M':
                is_signal = True
                side = 'Long'
                score = row['Day_Return_Pct']

            if is_signal:
                sl_factor = optimized_sl[mode].get(t, 0.5)
                
                candidates.append({
                    'ticker': t, 
                    'side': side, 
                    'score': score,
                    'sl_factor': sl_factor,
                    'ticker_group': ticker_group,
                    'current_vol': row['Prev_Volume'] / row['Prev_V_Avg'] if row['Prev_V_Avg'] > 0 else 1.0
                })
        
        sel = sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
        
        final_signals[mode] = [
            {
                'ticker': s['ticker'],
                'action': s['side'],
                'allocation_usd': ALLOCATION_USD,
                'sl_factor': round(s['sl_factor'], 2),
                'z_score': round(s['score'], 2) if mode != 'M' else None,
                'prev_return_pct': round(s['score'] * 100, 2) if mode == 'M' else None,
                'vol_ratio': round(s['current_vol'], 2),
                'ticker_group': s['ticker_group']
            }
            for s in sel
        ]
        
        print(f"  Kandidáti: {len(candidates)} → TOP 3:")
        for sig in final_signals[mode]:
            if mode == 'M':
                print(f"    • {sig['ticker']:6s} {sig['action']:5s} ({sig['ticker_group']:>10s}) | SL={sig['sl_factor']:.2f} | Ret={sig['prev_return_pct']:+.2f}%")
            else:
                print(f"    • {sig['ticker']:6s} {sig['action']:5s} ({sig['ticker_group']:>10s}) | SL={sig['sl_factor']:.2f} | Z={sig.get('z_score', 0):.2f}")
    
    return final_signals

def run_agent():
    """
    VEČERNÍ REŽIM (Evening-only):
    1. Vyhodnotí dnešní signály na 5min datech
    2. Uloží výsledky do CSV
    3. Přepočítá SL optimization
    4. Vygeneruje signály pro zítřek
    5. Uloží všechny soubory
    """
    print(f"🚀 TRADING AGENT - EVENING MODE")
    print(f"📅 Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # KROK 1: Vyhodnoť dnešní signály na 5min datech
    evaluated_trades = evaluate_todays_signals_on_5min_data()
    
    # KROK 2: Stáhni data pro generaci zítřejších signálů
    print(f"\n{'='*70}")
    print(f"📥 STAHOVÁNÍ DAT PRO GENERACI ZÍTŘEJŠÍCH SIGNÁLŮ")
    print(f"{'='*70}\n")
    
    try:
        raw = yf.download(ALL_TICKERS, period="8y", interval="1d", group_by='ticker', progress=False)
    except Exception as e:
        print(f"❌ Chyba při stahování dat: {e}")
        return
    
    ticker_data = {}
    for t in ALL_TICKERS:
        try:
            if t not in raw.columns.get_level_values(0):
                continue
            d = raw[t].dropna().copy()
            if len(d) < 100:
                continue
            
            d['Prev_High20_Strict'] = d['High'].rolling(window=20).max().shift(2)
            d['Prev_Low20_Strict'] = d['Low'].rolling(window=20).min().shift(2)
            d['Prev_Range'] = (d['High'] - d['Low']).shift(1)
            d['Prev_AvgRange'] = d['Prev_Range'].rolling(window=20).mean()
            d['Prev_Close'] = d['Close'].shift(1)
            d['Prev_Open'] = d['Open'].shift(1)
            d['Prev_Volume'] = d['Volume'].shift(1)
            d['Prev_V_Avg'] = d['Volume'].rolling(window=20).mean().shift(1)
            d['Prev_High'] = d['High'].shift(1)
            d['Day_Return_Pct'] = (d['Close'] - d['Open']) / d['Open']
            
            d_clean = d.dropna()
            if len(d_clean) >= 100:
                ticker_data[t] = d_clean
        except:
            continue
    
    print(f"✅ Načteno {len(ticker_data)} tickerů\n")
    
    # Grid search pro SL
    print(f"{'='*70}")
    print(f"🔍 GRID SEARCH PRO OPTIMÁLNÍ STOP LOSS")
    print(f"{'='*70}\n")
    
    ticker_performance = {m: {} for m in ['A', 'B', 'V', 'M']}
    optimized_sl = {m: {} for m in ['A', 'B', 'V', 'M']}
    optimization_results = {m: {} for m in ['A', 'B', 'V', 'M']}
    
    for mode_strat in ['A', 'B', 'V', 'M']:
        allowed_tickers = STRATEGY_TICKER_GROUPS[mode_strat]
        
        for t in allowed_tickers:
            if t not in ticker_data:
                continue
            df = ticker_data[t]
            hist_df = df.iloc[:-65]
            
            if mode_strat == 'A':
                sig_df = hist_df[abs(hist_df['Prev_Close'] - hist_df['Prev_High20_Strict']) / hist_df['Prev_AvgRange'] < 0.4]
            elif mode_strat == 'B':
                sig_df = hist_df[hist_df['Prev_Volume'] > hist_df['Prev_V_Avg'] * 1.5]
            elif mode_strat == 'V':
                sig_df = hist_df[hist_df['Prev_High'] > hist_df['Prev_High20_Strict']]
            elif mode_strat == 'M':
                sig_df = hist_df.copy()
            
            if len(sig_df) < 3:
                ticker_performance[mode_strat][t] = 0
                optimized_sl[mode_strat][t] = 0.5
                continue
            
            best_sl, best_metrics = optimize_sl_for_ticker_strategy(sig_df, mode_strat, t)
            
            if mode_strat == 'M':
                z_score = np.mean(sig_df['Day_Return_Pct'])
            else:
                z_score = calculate_z_score(sig_df['Day_Return_Pct'])
            
            ticker_performance[mode_strat][t] = z_score
            optimized_sl[mode_strat][t] = best_sl
            optimization_results[mode_strat][t] = best_metrics
    
    # Ulož optimization
    try:
        with open(OPTIMIZATION_FILE, 'w') as f:
            json.dump(optimization_results, f, indent=4)
        print(f"\n✅ Optimization uloženy\n")
    except Exception as e:
        print(f"❌ Chyba: {e}\n")
    
    # Backtest (minimal)
    backtest_results = run_backtest_60d(ticker_data, optimized_sl, ticker_performance)
    
    try:
        with open(BACKTEST_FILE, 'w') as f:
            json.dump(backtest_results, f, indent=4)
        print(f"✅ Backtest uloženy\n")
    except Exception as e:
        print(f"❌ Chyba: {e}\n")
    
    # Generuj signály pro zítřek
    final_signals = generate_signals_for_tomorrow(ticker_data, optimized_sl, ticker_performance)
    
    try:
        with open(SIGNAL_FILE, 'w') as f:
            json.dump(final_signals, f, indent=4)
        print(f"\n💾 Signály pro zítřek uloženy do: {SIGNAL_FILE}\n")
    except Exception as e:
        print(f"❌ Chyba: {e}\n")
    
    print(f"{'='*70}")
    print("✅ AGENT DOKONČEN!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_agent()
