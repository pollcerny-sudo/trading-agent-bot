import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
import traceback

# ────────────────────────────────────────────────
# Custom JSON encoder – řeší problém s pd.Timestamp / datetime
# ────────────────────────────────────────────────
class PandasJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)


# === MANUAL TICKER GROUPING ===
# Split tickers into two groups based on typical market cap/volatility
BIG_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]  # Large, stable companies
SMALL_TICKERS = ["TSLA", "AMD", "PLTR", "NFLX", "AVGO", "SMCI"]  # Smaller, more volatile
ETF_TICKERS = ["SPY", "QQQ"]  # ETFs can be in either group

# Combine for download
ALL_TICKERS = BIG_TICKERS + SMALL_TICKERS + ETF_TICKERS

# Strategy preferences (manual assignment)
STRATEGY_TICKER_GROUPS = {
    'A': BIG_TICKERS + ETF_TICKERS,   # Mean Reversion: prefers stable stocks
    'B': BIG_TICKERS + ETF_TICKERS,   # Volume Breakout: needs reliable volume
    'V': SMALL_TICKERS,               # Trend Breakout: benefits from volatility
    'M': SMALL_TICKERS                # Momentum: stronger in smaller caps
}

# Output files – relativní cesty → aktuální pracovní adresář
LOG_FILE = 'final_backtest_results.csv'
SIGNAL_FILE = 'ibkr_signals.json'
OPTIMIZATION_FILE = 'sl_optimization_results.json'
BACKTEST_FILE = 'backtest_60d_results.json'

COMMISSION_PCT = 0.001

# Grid search parametry
SL_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ALLOCATION_USD = 10000
BACKTEST_DAYS = 60


def calculate_z_score(profits):
    """Vypočítá Z-score pro sérii profitů"""
    if len(profits) < 3: return 0
    mean = np.mean(profits)
    std = np.std(profits, ddof=1)
    return mean / (std + 1e-9)


def simulate_trade_with_sl(row, side, sl_factor, commission_pct=0.001):
    """Simuluje obchod s stop lossem"""
    entry_price = row['Open']
    exit_price = row['Close']
    sl_distance = sl_factor * row['Prev_AvgRange']
    shares = int(ALLOCATION_USD / entry_price) if entry_price > 0 else 1
    
    hit_sl = False
    
    if side == 'Long':
        sl_price = entry_price - sl_distance
        if row['Low'] <= sl_price:
            exit_price = sl_price
            hit_sl = True
        gross_pnl = shares * (exit_price - entry_price)
    else:  # Short
        sl_price = entry_price + sl_distance
        if row['High'] >= sl_price:
            exit_price = sl_price
            hit_sl = True
        gross_pnl = shares * (entry_price - exit_price)
    
    commission = ALLOCATION_USD * commission_pct * 2
    net_pnl = gross_pnl - commission
    
    return net_pnl, hit_sl


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
            'win_rate': sum(1 for p in profits if p > 0) / len(profits) * 100 if profits else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'sharpe': calculate_z_score(profits),
            'num_trades': len(trades),
            'sl_hit_rate': sum(1 for t in trades if t['hit_sl']) / len(trades) * 100 if trades else 0
        })
    
    if not results:
        return 0.5, None
    
    best = max(results, key=lambda x: x['sharpe'])
    return best['sl_factor'], best


def calculate_max_drawdown(equity_curve):
    """Vypočítá maximální drawdown v procentech"""
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


def run_backtest_60d(ticker_data, optimized_sl, ticker_performance):
    """Spustí 60denní backtest s manuálně přiřazenými ticker skupinami"""
    print(f"\n{'='*70}")
    print(f"📊 BACKTEST POSLEDNÍCH {BACKTEST_DAYS} DNÍ - TICKER GROUP ASSIGNMENT")
    print(f"{'='*70}\n")
    
    backtest_results = {}
    
    for mode in ['A', 'B', 'V', 'M']:
        print(f"\n🎯 Strategie {mode}:")
        print("-" * 70)
        
        allowed_tickers = STRATEGY_TICKER_GROUPS[mode]
        filtered_data = {t: ticker_data[t] for t in allowed_tickers if t in ticker_data}
        
        if not filtered_data:
            print(f"  ⚠️  Žádné tickery k dispozici")
            backtest_results[mode] = {'total_profit': 0, 'num_trades': 0, 'message': 'Žádné tickery'}
            continue
        
        print(f"  📋 Použité tickery ({len(filtered_data)}): {', '.join(filtered_data.keys())}")
        
        daily_trades = []
        equity_curve = [10000]
        
        for day_offset in range(BACKTEST_DAYS, 0, -1):
            candidates = []
            
            for t in filtered_data:
                df = ticker_data[t]
                
                if len(df) < day_offset + 1:
                    continue
                
                row = df.iloc[-(day_offset + 1)]
                current_row = df.iloc[-day_offset]
                
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
                        'row': current_row
                    })
            
            top3 = sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
            
            day_pnl = 0
            for trade in top3:
                profit, hit_sl = simulate_trade_with_sl(
                    trade['row'], 
                    trade['side'], 
                    trade['sl_factor'], 
                    COMMISSION_PCT
                )
                day_pnl += profit
                
                daily_trades.append({
                    'date': trade['row'].name,
                    'ticker': trade['ticker'],
                    'side': trade['side'],
                    'profit': profit,
                    'hit_sl': hit_sl,
                    'score': trade['score'],
                    'sl_factor': trade['sl_factor']
                })
            
            if top3:
                equity_curve.append(equity_curve[-1] + day_pnl)
        
        # Výpočet statistik
        if daily_trades:
            profits = [t['profit'] for t in daily_trades]
            total_profit = sum(profits)
            total_return_pct = (equity_curve[-1] / equity_curve[0] - 1) * 100
            num_trades = len(daily_trades)
            win_rate = sum(1 for p in profits if p > 0) / num_trades * 100
            avg_win = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
            avg_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
            sharpe = calculate_z_score(profits)
            max_dd = calculate_max_drawdown(equity_curve)
            sl_hit_rate = sum(1 for t in daily_trades if t['hit_sl']) / num_trades * 100
            
            daily_pnls = {}
            for t in daily_trades:
                date_str = t['date'].strftime('%Y-%m-%d')
                if date_str not in daily_pnls:
                    daily_pnls[date_str] = 0
                daily_pnls[date_str] += t['profit']
            
            winning_days = sum(1 for pnl in daily_pnls.values() if pnl > 0)
            total_days = len(daily_pnls)
            daily_win_rate = winning_days / total_days * 100 if total_days > 0 else 0
            
            backtest_results[mode] = {
                'total_profit': round(total_profit, 2),
                'total_return_pct': round(total_return_pct, 2),
                'final_equity': round(equity_curve[-1], 2),
                'num_trades': num_trades,
                'num_days': total_days,
                'win_rate': round(win_rate, 2),
                'daily_win_rate': round(daily_win_rate, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'sharpe_ratio': round(sharpe, 2),
                'max_drawdown_pct': round(max_dd, 2),
                'sl_hit_rate': round(sl_hit_rate, 2),
                'equity_curve': [round(e, 2) for e in equity_curve],
                'trades': daily_trades[-10:],
                'tickers_used': list(filtered_data.keys()),
                'ticker_group': 'BIG' if mode in ['A', 'B'] else 'SMALL'
            }
            
            print(f"  💰 Celkový Profit:        ${total_profit:,.2f} ({total_return_pct:+.2f}%)")
            print(f"  📈 Finální Equity:        ${equity_curve[-1]:,.2f}")
            print(f"  📊 Počet Obchodů:         {num_trades} ({total_days} dnů)")
            print(f"  ✅ Win Rate (obchody):    {win_rate:.1f}%")
            print(f"  📅 Win Rate (dny):        {daily_win_rate:.1f}%")
            print(f"  💚 Průměrný Win:          ${avg_win:.2f}")
            print(f"  💔 Průměrný Loss:         ${avg_loss:.2f}")
            print(f"  📉 Max Drawdown:          {max_dd:.2f}%")
            print(f"  🎲 Sharpe Ratio:          {sharpe:.2f}")
            print(f"  🛑 Stop Loss Hit Rate:    {sl_hit_rate:.1f}%")
        else:
            print(f"  ⚠️  Žádné obchody v backtestu")
            backtest_results[mode] = {
                'total_profit': 0,
                'num_trades': 0,
                'message': 'Žádné signály',
                'tickers_used': list(filtered_data.keys()),
                'ticker_group': 'BIG' if mode in ['A', 'B'] else 'SMALL'
            }
    
    print(f"\n{'='*70}\n")
    return backtest_results


def run_agent():
    print(f"🚀 TRADING AGENT S MANUÁLNÍ TICKER SEGMENTACÍ")
    print(f"📅 Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Kontrola zapisovatelnosti
    cwd = os.getcwd()
    print(f"📁 Pracovní adresář: {cwd}")
    print(f"   Zapisovatelný: {os.access(cwd, os.W_OK)}\n")
    
    print(f"📋 TICKER SKUPINY:")
    print(f"   BIG TICKERS (stable):   {', '.join(BIG_TICKERS)}")
    print(f"   SMALL TICKERS (volatile): {', '.join(SMALL_TICKERS)}")
    print(f"   ETF TICKERS:            {', '.join(ETF_TICKERS)}\n")
    
    print(f"📋 STRATEGY → TICKER GROUP ASSIGNMENT:")
    print(f"   A (Mean Reversion):   → BIG + ETF")
    print(f"   B (Volume Breakout):  → BIG + ETF")
    print(f"   V (Trend Breakout):   → SMALL")
    print(f"   M (Momentum):         → SMALL\n")
    
    print(f"📁 VÝSTUPNÍ SOUBORY:")
    print(f"   {LOG_FILE}")
    print(f"   {SIGNAL_FILE}")
    print(f"   {OPTIMIZATION_FILE}")
    print(f"   {BACKTEST_FILE}\n")
    
    # 1. Stažení dat
    print(f"📥 Stahuji data pro {len(ALL_TICKERS)} tickerů...")
    try:
        raw = yf.download(ALL_TICKERS, period="2y", interval="1d", group_by='ticker', progress=False)
    except Exception as e:
        print(f"❌ Chyba při stahování dat: {e}")
        return
    
    ticker_data = {}
    failed_tickers = []
    
    for t in ALL_TICKERS:
        try:
            if t not in raw.columns.get_level_values(0):
                failed_tickers.append(t)
                print(f"⚠️  Ticker {t}: Žádná data nenalezena")
                continue
                
            d = raw[t].dropna().copy()
            
            if len(d) < 100:
                failed_tickers.append(t)
                print(f"⚠️  Ticker {t}: Nedostatek dat ({len(d)} dní)")
                continue
            
            # Technické indikátory
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
            
            if len(d_clean) < 100:
                failed_tickers.append(t)
                print(f"⚠️  Ticker {t}: Nedostatek validních dat po výpočtech")
                continue
                
            ticker_data[t] = d_clean
            print(f"✅ Ticker {t}: Načteno {len(d_clean)} dní dat")
            
        except Exception as e:
            failed_tickers.append(t)
            print(f"❌ Ticker {t}: Chyba - {str(e)}")
            continue
    
    if not ticker_data:
        print("❌ Žádná validní data. Ukončuji.")
        return
    
    print(f"\n📊 Úspěšně načteno: {len(ticker_data)} tickerů")
    if failed_tickers:
        print(f"⚠️  Selhalo: {', '.join(failed_tickers)}")

    # 2. GRID SEARCH PRO STOP LOSS
    print(f"\n{'='*70}")
    print(f"🔍 GRID SEARCH PRO OPTIMÁLNÍ STOP LOSS")
    print(f"{'='*70}")
    print(f"   Testované SL faktory: {SL_GRID}\n")
    
    ticker_performance = {m: {} for m in ['A', 'B', 'V', 'M']}
    optimized_sl = {m: {} for m in ['A', 'B', 'V', 'M']}
    optimization_results = {m: {} for m in ['A', 'B', 'V', 'M']}
    
    for mode in ['A', 'B', 'V', 'M']:
        allowed_tickers = STRATEGY_TICKER_GROUPS[mode]
        ticker_group = 'BIG + ETF' if mode in ['A', 'B'] else 'SMALL'
        
        print(f"\n🎯 Strategie {mode} ({ticker_group}):")
        print(f"   Tickery: {', '.join([t for t in allowed_tickers if t in ticker_data])}\n")
        
        for t in allowed_tickers:
            if t not in ticker_data:
                continue
                
            df = ticker_data[t]
            hist_df = df.iloc[:-65]
            
            # Filtrace podle strategie
            if mode == 'A':
                sig_df = hist_df[abs(hist_df['Prev_Close'] - hist_df['Prev_High20_Strict']) / hist_df['Prev_AvgRange'] < 0.4]
            elif mode == 'B':
                sig_df = hist_df[hist_df['Prev_Volume'] > hist_df['Prev_V_Avg'] * 1.5]
            elif mode == 'V':
                sig_df = hist_df[hist_df['Prev_High'] > hist_df['Prev_High20_Strict']]
            elif mode == 'M':
                sig_df = hist_df.copy()
            
            if len(sig_df) < 3:
                ticker_performance[mode][t] = 0
                optimized_sl[mode][t] = 0.5
                continue
            
            best_sl, best_metrics = optimize_sl_for_ticker_strategy(sig_df, mode, t)
            
            if mode == 'M':
                z_score = np.mean(sig_df['Day_Return_Pct'])
            else:
                z_score = calculate_z_score(sig_df['Day_Return_Pct'])
            
            ticker_performance[mode][t] = z_score
            optimized_sl[mode][t] = best_sl
            optimization_results[mode][t] = best_metrics
            
            if best_metrics:
                print(f"  {t:6s}: SL={best_sl:.1f} | Sharpe={best_metrics['sharpe']:.2f} | WR={best_metrics['win_rate']:.0f}%")

    # Uložení optimization výsledků
    try:
        print(f"\n💾 Ukládám optimization výsledky do {OPTIMIZATION_FILE} ...")
        total_records = sum(len(optimization_results[m]) for m in optimization_results)
        print(f"   Celkem záznamů: {total_records}")
        
        with open(OPTIMIZATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, indent=4, cls=PandasJSONEncoder)
        
        print(f"✅ Optimization výsledky uloženy ({os.path.getsize(OPTIMIZATION_FILE):,} bytes)\n")
    except Exception as e:
        print(f"❌ Chyba při ukládání {OPTIMIZATION_FILE}: {e}")
        traceback.print_exc()
        print()

    # 3. BACKTEST
    backtest_results = run_backtest_60d(ticker_data, optimized_sl, ticker_performance)
    
    # Uložení backtest výsledků
    try:
        print(f"💾 Ukládám backtest výsledky do {BACKTEST_FILE} ...")
        
        results_to_save = {}
        for mode, data in backtest_results.items():
            results_to_save[mode] = data.copy()
            if 'trades' in results_to_save[mode]:
                for trade in results_to_save[mode]['trades']:
                    if 'date' in trade and hasattr(trade['date'], 'strftime'):
                        trade['date'] = trade['date'].strftime('%Y-%m-%d')
        
        with open(BACKTEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, cls=PandasJSONEncoder)
        
        print(f"✅ Backtest výsledky uloženy ({os.path.getsize(BACKTEST_FILE):,} bytes)\n")
    except Exception as e:
        print(f"❌ Chyba při ukládání {BACKTEST_FILE}: {e}")
        traceback.print_exc()
        print()

    # 4. GENERACE SIGNÁLŮ PRO DNES
    print(f"{'='*70}")
    print(f"🎯 GENERACE SIGNÁLŮ PRO DNES")
    print(f"{'='*70}\n")
    
    final_signals = {}
    eval_logs = []
    
    for mode in ['A', 'B', 'V', 'M']:
        allowed_tickers = STRATEGY_TICKER_GROUPS[mode]
        filtered_data = {t: ticker_data[t] for t in allowed_tickers if t in ticker_data}
        ticker_group = 'BIG + ETF' if mode in ['A', 'B'] else 'SMALL'
        
        print(f"\nStrategie {mode} ({ticker_group}):")
        print(f"  Allowed tickery: {', '.join(allowed_tickers)}")
        
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
        
        for s in sel:
            d_row = ticker_data[s['ticker']].iloc[-1]
            profit, hit_sl = simulate_trade_with_sl(d_row, s['side'], s['sl_factor'], COMMISSION_PCT)
            
            eval_logs.append({
                'Date': d_row.name.strftime('%Y-%m-%d'),
                'Strategy': mode,
                'Ticker': s['ticker'],
                'Side': s['side'],
                'Type': 'EOD',
                'Profit': round(profit, 2),
                'Score': round(s['score'], 4),
                'SL-Factor': round(s['sl_factor'], 2),
                'Ticker-Group': s['ticker_group'],
                'Hit-SL': hit_sl
            })

    # Uložení signálů
    try:
        with open(SIGNAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_signals, f, indent=4, cls=PandasJSONEncoder)
        print(f"\n💾 Signály uloženy do: {SIGNAL_FILE}\n")
    except Exception as e:
        print(f"❌ Chyba při ukládání signálů: {e}\n")
        
    if eval_logs:
        try:
            df_new = pd.DataFrame(eval_logs)
            df_new.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
            print(f"✅ {len(eval_logs)} obchodů zalogováno do: {LOG_FILE}\n")
        except Exception as e:
            print(f"❌ Chyba při ukládání logu: {e}\n")
    
    # Shrnutí
    print(f"{'='*70}")
    print(f"📊 SHRNUTÍ S TICKER GROUP SEGMENTACÍ")
    print(f"{'='*70}\n")
    
    for mode in ['A', 'B', 'V', 'M']:
        if mode in backtest_results and 'total_profit' in backtest_results[mode]:
            result = backtest_results[mode]
            tickers = result.get('tickers_used', [])
            group = result.get('ticker_group', 'N/A')
            print(f"{mode} ({group}) - Tickery: {', '.join(tickers)}")
            print(f"  Profit: ${result['total_profit']:>10,.2f} ({result['total_return_pct']:>+6.2f}%) | Sharpe: {result['sharpe_ratio']:>5.2f}")
    
    print(f"\n{'='*70}")
    print("✅ AGENT DOKONČEN!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_agent()
