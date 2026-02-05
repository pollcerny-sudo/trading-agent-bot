import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === KONFIGURACE ===
GITHUB_BASE = 'https://raw.githubusercontent.com/pollcerny-sudo/trading-agent-bot/main/'
CSV_URL = GITHUB_BASE + 'final_backtest_results.csv'
JSON_URL = GITHUB_BASE + 'ibkr_signals.json'
BACKTEST_URL = GITHUB_BASE + 'backtest_60d_results.json'

# Barvy pro strategie
STRATEGY_COLORS = {
    'A': '#2E86AB',  # Modrá
    'B': '#A23B72',  # Fialová
    'V': '#F18F01',  # Oranžová
    'M': '#06A77D'   # Zelená
}

STRATEGY_NAMES = {
    'A': 'Mean Reversion',
    'B': 'Volume Breakout',
    'V': 'Trend Breakout',
    'M': 'Momentum'
}

def download_data():
    """Stáhne všechna data z GitHubu"""
    print(f"{'='*80}")
    print(f"📥 STAHOVÁNÍ DAT Z GITHUBU")
    print(f"{'='*80}\n")

    data = {}

    # 1. CSV s historií obchodů
    try:
        print(f"📄 Stahuji CSV historii obchodů...")
        df = pd.read_csv(CSV_URL)
        data['history'] = df
        print(f"   ✅ Načteno {len(df)} obchodů od {df['Date'].min()} do {df['Date'].max()}")
    except Exception as e:
        print(f"   ❌ Chyba: {e}")
        data['history'] = pd.DataFrame()

    # 2. JSON s dnešními signály
    try:
        print(f"📄 Stahuji dnešní signály (JSON)...")
        response = requests.get(JSON_URL, timeout=10)
        response.raise_for_status()
        signals = response.json()
        data['signals'] = signals
        total_signals = sum(len(signals.get(s, [])) for s in ['A', 'B', 'V', 'M'])
        print(f"   ✅ Načteno {total_signals} signálů pro dnes")
    except Exception as e:
        print(f"   ❌ Chyba: {e}")
        data['signals'] = {}

    # 3. JSON s výsledky 60d backtestů
    try:
        print(f"📄 Stahuji výsledky 60d backtestu (JSON)...")
        response = requests.get(BACKTEST_URL, timeout=10)
        response.raise_for_status()
        backtest = response.json()
        data['backtest'] = backtest
        print(f"   ✅ Načteny backtest výsledky pro {len(backtest)} strategií")
    except Exception as e:
        print(f"   ❌ Chyba: {e}")
        data['backtest'] = {}

    print(f"\n{'='*80}\n")
    return data

def calculate_metrics(df):
    """Vypočítá metriky pro celé portfolio nebo strategii"""
    if df.empty:
        return None

    df = df.copy()
    df['Cumulative_Profit'] = df['Profit'].cumsum()

    total_profit = df['Profit'].sum()
    num_trades = len(df)
    win_rate = (df['Profit'] > 0).sum() / num_trades * 100 if num_trades > 0 else 0

    avg_win = df[df['Profit'] > 0]['Profit'].mean() if (df['Profit'] > 0).any() else 0
    avg_loss = df[df['Profit'] < 0]['Profit'].mean() if (df['Profit'] < 0).any() else 0

    # Sharpe ratio
    if len(df) > 1:
        returns = df['Profit'].values
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    equity = 10000 + df['Cumulative_Profit'].values
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_dd = drawdown.max() if len(drawdown) > 0 else 0

    return {
        'total_profit': total_profit,
        'total_return_pct': (total_profit / 10000) * 100,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
    }

def compare_backtest_vs_actual(data):
    """Porovná backtest předpověď vs skutečné výsledky"""
    print(f"{'='*80}")
    print(f"🔍 POROVNÁNÍ BACKTEST vs SKUTEČNOST")
    print(f"{'='*80}\n")

    if data['history'].empty or not data['backtest']:
        print("⚠️  Nedostatek dat pro porovnání\n")
        return

    # Použijeme poslední 60 dní z CSV jako "skutečnost"
    df = data['history'].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Posledních 60 dní
    cutoff_date = df['Date'].max() - timedelta(days=60)
    recent_df = df[df['Date'] > cutoff_date]

    print(f"📊 Analýza období: {recent_df['Date'].min().strftime('%Y-%m-%d')} až {recent_df['Date'].max().strftime('%Y-%m-%d')}\n")

    comparison = []

    for strategy in ['A', 'B', 'V', 'M']:
        # Backtest předpověď
        backtest_result = data['backtest'].get(strategy, {})

        # Skutečné výsledky
        actual_df = recent_df[recent_df['Strategy'] == strategy]
        actual_metrics = calculate_metrics(actual_df) if not actual_df.empty else None

        if backtest_result and actual_metrics:
            bt_profit = backtest_result.get('total_profit', 0)
            bt_return = backtest_result.get('total_return_pct', 0)
            bt_sharpe = backtest_result.get('sharpe_ratio', 0)
            bt_trades = backtest_result.get('num_trades', 0)

            actual_profit = actual_metrics['total_profit']
            actual_return = actual_metrics['total_return_pct']
            actual_sharpe = actual_metrics['sharpe_ratio']
            actual_trades = actual_metrics['num_trades']

            # Rozdíly
            profit_diff = actual_profit - bt_profit
            profit_diff_pct = (profit_diff / abs(bt_profit) * 100) if bt_profit != 0 else 0

            comparison.append({
                'Strategy': f"{strategy} - {STRATEGY_NAMES[strategy]}",
                'Backtest_Profit': bt_profit,
                'Actual_Profit': actual_profit,
                'Difference': profit_diff,
                'Diff_Pct': profit_diff_pct,
                'BT_Sharpe': bt_sharpe,
                'Actual_Sharpe': actual_sharpe,
                'BT_Trades': bt_trades,
                'Actual_Trades': actual_trades
            })

            print(f"Strategie {strategy} - {STRATEGY_NAMES[strategy]}:")
            print(f"  Backtest:  Profit=${bt_profit:>10,.2f} | Return={bt_return:>+6.2f}% | Sharpe={bt_sharpe:>5.2f} | Trades={bt_trades:>4}")
            print(f"  Skutečné:  Profit=${actual_profit:>10,.2f} | Return={actual_return:>+6.2f}% | Sharpe={actual_sharpe:>5.2f} | Trades={actual_trades:>4}")
            print(f"  Rozdíl:    ${profit_diff:>+10,.2f} ({profit_diff_pct:>+6.1f}%)")

            # Hodnocení přesnosti
            if abs(profit_diff_pct) < 10:
                print(f"  ✅ Velmi přesný backtest!")
            elif abs(profit_diff_pct) < 25:
                print(f"  ✓ Přijatelná přesnost")
            else:
                print(f"  ⚠️  Velká odchylka od backtestů")
            print()

    if comparison:
        comp_df = pd.DataFrame(comparison)
        print(f"\n📈 Průměrná odchylka backtestu: {comp_df['Diff_Pct'].abs().mean():.1f}%")

    print(f"{'='*80}\n")
    return comparison

def plot_equity_curves(data):
    """Vykreslí equity křivky pro všechny strategie"""
    print(f"📊 Generuji Equity Curves...\n")

    if data['history'].empty:
        print("⚠️  Žádná data pro equity křivky\n")
        return

    df = data['history'].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('🚀 EQUITY CURVES - POROVNÁNÍ STRATEGIÍ', fontsize=16, fontweight='bold')

    # 1. Celková equity křivka (všechny strategie)
    ax1 = axes[0, 0]
    for strategy in ['A', 'B', 'V', 'M']:
        strat_df = df[df['Strategy'] == strategy].copy()
        if not strat_df.empty:
            strat_df = strat_df.sort_values('Date')
            strat_df['Equity'] = 10000 + strat_df['Profit'].cumsum()
            ax1.plot(strat_df['Equity'].values,
                    label=f"{strategy} - {STRATEGY_NAMES[strategy]}",
                    linewidth=2.5,
                    color=STRATEGY_COLORS[strategy])

    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Start Capital')
    ax1.set_title('Equity Curves - Všechny Strategie', fontweight='bold')
    ax1.set_xlabel('Počet obchodů')
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Kumulativní profit v čase (denní agregace)
    ax2 = axes[0, 1]
    daily_profits = df.groupby(['Date', 'Strategy'])['Profit'].sum().reset_index()

    for strategy in ['A', 'B', 'V', 'M']:
        strat_daily = daily_profits[daily_profits['Strategy'] == strategy].copy()
        if not strat_daily.empty:
            strat_daily = strat_daily.sort_values('Date')
            strat_daily['Cumulative'] = strat_daily['Profit'].cumsum()
            ax2.plot(strat_daily['Date'],
                    strat_daily['Cumulative'],
                    label=f"{strategy} - {STRATEGY_NAMES[strategy]}",
                    linewidth=2.5,
                    color=STRATEGY_COLORS[strategy])

    ax2.set_title('Kumulativní Profit v Čase', fontweight='bold')
    ax2.set_xlabel('Datum')
    ax2.set_ylabel('Kumulativní Profit ($)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. Backtest vs Actual equity křivky (pokud máme backtest data)
    ax3 = axes[1, 0]
    if data['backtest']:
        for strategy in ['A', 'B', 'V', 'M']:
            bt = data['backtest'].get(strategy, {})
            if 'equity_curve' in bt:
                equity_curve = bt['equity_curve']
                ax3.plot(equity_curve,
                        label=f"{strategy} - Backtest",
                        linewidth=2,
                        linestyle='--',
                        alpha=0.7,
                        color=STRATEGY_COLORS[strategy])

        ax3.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('Backtest Equity Curves (60 dní)', fontweight='bold')
        ax3.set_xlabel('Dny')
        ax3.set_ylabel('Equity ($)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Backtest data nedostupná',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Backtest Equity Curves (60 dní)', fontweight='bold')

    # 4. Drawdown analýza
    ax4 = axes[1, 1]
    for strategy in ['A', 'B', 'V', 'M']:
        strat_df = df[df['Strategy'] == strategy].copy()
        if not strat_df.empty:
            strat_df = strat_df.sort_values('Date')
            equity = 10000 + strat_df['Profit'].cumsum().values
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak * 100
            ax4.plot(drawdown,
                    label=f"{strategy} - {STRATEGY_NAMES[strategy]}",
                    linewidth=2,
                    color=STRATEGY_COLORS[strategy])

    ax4.set_title('Drawdown Analysis (%)', fontweight='bold')
    ax4.set_xlabel('Počet obchodů')
    ax4.set_ylabel('Drawdown (%)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig('dashboard_equity_curves.png', dpi=150, bbox_inches='tight')
    print("✅ Graf uložen jako 'dashboard_equity_curves.png'\n")
    plt.show()

def plot_strategy_comparison(data):
    """Vytvoří srovnávací metriky pro strategie"""
    print(f"📊 Generuji Strategy Comparison Matrix...\n")

    if data['history'].empty:
        print("⚠️  Žádná data pro porovnání\n")
        return

    df = data['history'].copy()

    # Výpočet metrik pro každou strategii
    metrics_data = []
    for strategy in ['A', 'B', 'V', 'M']:
        strat_df = df[df['Strategy'] == strategy]
        if not strat_df.empty:
            metrics = calculate_metrics(strat_df)
            metrics['Strategy'] = f"{strategy}\n{STRATEGY_NAMES[strategy]}"
            metrics_data.append(metrics)

    if not metrics_data:
        print("⚠️  Žádné metriky k zobrazení\n")
        return

    metrics_df = pd.DataFrame(metrics_data)

    # Vytvoř 2x3 grid grafů
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('📊 STRATEGY COMPARISON - KEY METRICS', fontsize=16, fontweight='bold')

    strategies = metrics_df['Strategy'].values
    colors = [STRATEGY_COLORS[s.split('\n')[0]] for s in strategies]

    # 1. Total Profit
    ax1 = axes[0, 0]
    bars1 = ax1.bar(strategies, metrics_df['total_profit'], color=colors, alpha=0.8)
    ax1.set_title('Total Profit ($)', fontweight='bold')
    ax1.set_ylabel('Profit ($)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom' if height > 0 else 'top')

    # 2. Win Rate
    ax2 = axes[0, 1]
    bars2 = ax2.bar(strategies, metrics_df['win_rate'], color=colors, alpha=0.8)
    ax2.set_title('Win Rate (%)', fontweight='bold')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

    # 3. Sharpe Ratio
    ax3 = axes[0, 2]
    bars3 = ax3.bar(strategies, metrics_df['sharpe_ratio'], color=colors, alpha=0.8)
    ax3.set_title('Sharpe Ratio', fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1.0')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')

    # 4. Max Drawdown
    ax4 = axes[1, 0]
    bars4 = ax4.bar(strategies, metrics_df['max_drawdown_pct'], color=colors, alpha=0.8)
    ax4.set_title('Max Drawdown (%)', fontweight='bold')
    ax4.set_ylabel('Drawdown (%)')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

    # 5. Number of Trades
    ax5 = axes[1, 1]
    bars5 = ax5.bar(strategies, metrics_df['num_trades'], color=colors, alpha=0.8)
    ax5.set_title('Number of Trades', fontweight='bold')
    ax5.set_ylabel('Trades')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

    # 6. Profit Factor
    ax6 = axes[1, 2]
    bars6 = ax6.bar(strategies, metrics_df['profit_factor'], color=colors, alpha=0.8)
    ax6.set_title('Profit Factor (Avg Win / Avg Loss)', fontweight='bold')
    ax6.set_ylabel('Profit Factor')
    ax6.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1.0')
    ax6.grid(True, alpha=0.3, axis='y')
    for bar in bars6:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('dashboard_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Graf uložen jako 'dashboard_comparison.png'\n")
    plt.show()

def show_today_signals(data):
    """Zobrazí dnešní signály"""
    print(f"{'='*80}")
    print(f"🎯 DNEŠNÍ SIGNÁLY")
    print(f"{'='*80}\n")

    if not data['signals']:
        print("⚠️  Žádné signály pro dnešek\n")
        return

    for strategy in ['A', 'B', 'V', 'M']:
        signals = data['signals'].get(strategy, [])
        print(f"\n{strategy} - {STRATEGY_NAMES[strategy]}:")
        print("-" * 80)

        if not signals:
            print("   Žádné signály")
        else:
            for sig in signals:
                ticker = sig.get('ticker', 'N/A')
                action = sig.get('action', 'N/A')
                allocation = sig.get('allocation_usd', 0)
                sl_factor = sig.get('sl_factor', 0)

                if strategy == 'M':
                    prev_return = sig.get('prev_return_pct', 0)
                    print(f"   • {ticker:6s} | {action:5s} | Alloc: ${allocation:>6,} | "
                          f"SL: {sl_factor:.2f} | Prev Return: {prev_return:>+6.2f}%")
                else:
                    z_score = sig.get('z_score', 0)
                    print(f"   • {ticker:6s} | {action:5s} | Alloc: ${allocation:>6,} | "
                          f"SL: {sl_factor:.2f} | Z-Score: {z_score:>6.2f}")

    print(f"\n{'='*80}\n")

def show_summary_table(data):
    """Zobrazí souhrnnou tabulku všech metrik"""
    print(f"{'='*80}")
    print(f"📋 SOUHRNNÁ TABULKA - VŠECHNY STRATEGIE")
    print(f"{'='*80}\n")

    if data['history'].empty:
        print("⚠️  Žádná data pro tabulku\n")
        return

    df = data['history'].copy()

    summary_data = []
    for strategy in ['A', 'B', 'V', 'M']:
        strat_df = df[df['Strategy'] == strategy]
        if not strat_df.empty:
            metrics = calculate_metrics(strat_df)
            summary_data.append({
                'Strategy': f"{strategy} - {STRATEGY_NAMES[strategy]}",
                'Total Profit': f"${metrics['total_profit']:,.2f}",
                'Return %': f"{metrics['total_return_pct']:+.2f}%",
                'Trades': metrics['num_trades'],
                'Win Rate': f"{metrics['win_rate']:.1f}%",
                'Avg Win': f"${metrics['avg_win']:.2f}",
                'Avg Loss': f"${metrics['avg_loss']:.2f}",
                'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                'Max DD': f"{metrics['max_drawdown_pct']:.2f}%",
                'PF': f"{metrics['profit_factor']:.2f}"
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print()

        # Najdi nejlepší strategii podle různých metrik
        df_numeric = data['history'].copy()
        best_by_profit = df_numeric.groupby('Strategy')['Profit'].sum().idxmax()
        best_by_sharpe = None

        for strategy in ['A', 'B', 'V', 'M']:
            strat_df = df[df['Strategy'] == strategy]
            if not strat_df.empty:
                metrics = calculate_metrics(strat_df)
                if best_by_sharpe is None or metrics['sharpe_ratio'] > calculate_metrics(df[df['Strategy'] == best_by_sharpe])['sharpe_ratio']:
                    best_by_sharpe = strategy

        print(f"🏆 Nejlepší strategie:")
        print(f"   Podle Profit:      {best_by_profit} - {STRATEGY_NAMES[best_by_profit]}")
        if best_by_sharpe:
            print(f"   Podle Sharpe:      {best_by_sharpe} - {STRATEGY_NAMES[best_by_sharpe]}")

    print(f"\n{'='*80}\n")


def show_recent_performance(data):
    """Zobrazí výkonnost za posledních 7, 30 a 60 dní"""
    print(f"{'='*80}")
    print(f"📅 VÝKONNOST ZA RŮZNÁ OBDOBÍ")
    print(f"{'='*80}\n")

    if data['history'].empty:
        print("⚠️  Žádná data\n")
        return

    df = data['history'].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    max_date = df['Date'].max()

    periods = [
        ('7 dní', 7),
        ('30 dní', 30),
        ('60 dní', 60)
    ]

    for period_name, days in periods:
        print(f"\n📊 Poslední {period_name}:")
        print("-" * 80)

        cutoff = max_date - timedelta(days=days)
        period_df = df[df['Date'] > cutoff]

        if period_df.empty:
            print(f"   Žádná data za tento období")
            continue

        for strategy in ['A', 'B', 'V', 'M']:
            strat_df = period_df[period_df['Strategy'] == strategy]
            if not strat_df.empty:
                profit = strat_df['Profit'].sum()
                trades = len(strat_df)
                win_rate = (strat_df['Profit'] > 0).sum() / trades * 100
                print(f"   {strategy} - {STRATEGY_NAMES[strategy]:20s}: "
                      f"Profit=${profit:>10,.2f} | Trades={trades:>4} | WR={win_rate:>5.1f}%")

    print(f"\n{'='*80}\n")

def export_dashboard_json(data):
    if data['history'].empty:
        return

    df = data['history'].copy()
    df['Date'] = pd.to_datetime(df['Date'])

    strategies = {}

    for s in ['A','B','V','M']:
        strat_df = df[df['Strategy']==s]
        if strat_df.empty:
            continue
        m = calculate_metrics(strat_df)
        strategies[s] = {
            "name": STRATEGY_NAMES[s],
            "metrics": m
        }

    df_daily = df.groupby(['Date','Strategy'])['Profit'].sum().reset_index()

    equity = {}

for s in ['A','B','V','M']:
    d = df_daily[df_daily['Strategy']==s].sort_values('Date')
    if d.empty:
        equity[s] = []
        continue

    d['Equity'] = 10000 + d['Profit'].cumsum()

    equity[s] = [
        {"date": str(r['Date'])[:10], "equity": round(r['Equity'],2)}
        for _, r in d.iterrows()
    ]
    import os, json
    os.makedirs("public", exist_ok=True)

    out = {
        "timestamp": datetime.now().isoformat(),
        "strategies": strategies,
        "signals": data['signals'],
        "equity": equity
    }

    with open("public/dashboard_data.json","w") as f:
        json.dump(out,f,indent=2)


def main():
    """Hlavní funkce dashboardu"""
    print(f"\n{'='*80}")
    print(f"🚀 TRADING AGENT DASHBOARD")
    print(f"{'='*80}")
    print(f"📅 Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # 1. Stažení dat
    data = download_data()

    # 2. Zobrazení dnešních signálů
    show_today_signals(data)

    # 3. Souhrnná tabulka
    show_summary_table(data)

    # 4. Výkonnost za různá období
    show_recent_performance(data)

    # 5. Porovnání backtest vs skutečnost
    compare_backtest_vs_actual(data)

    # 6. Equity křivky
    plot_equity_curves(data)

    # 7. Srovnání strategií
    plot_strategy_comparison(data)

    # ✅ 8. EXPORT JSON PRO WEB DASHBOARD
    export_dashboard_json(data)

    print(f"{'='*80}")
    print(f"✅ DASHBOARD DOKONČEN")
    print(f"{'='*80}")
    print(f"📁 Vygenerované soubory:")
    print(f"   • dashboard_equity_curves.png")
    print(f"   • dashboard_comparison.png")
    print(f"   • public/dashboard_data.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
