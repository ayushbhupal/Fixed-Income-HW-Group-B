import pandas as pd
import numpy as np

# --- Identify SOFR contracts ---
contracts = [c for c in sofr_rates.columns if c.startswith("SR3")]
month_map = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}

def parse_contract(c):
    month = month_map[c[3]]
    year = 2020 + int(c[4])   # adjust if your data goes beyond 2029
    return (year, month)

contracts_sorted = sorted(contracts, key=parse_contract)
sofr_rates = sofr_rates[contracts_sorted].copy()

# --- 2nd-contract carry strategy ---
held_contract = []
current_contract = None

for date, row in sofr_rates.iterrows():
    # Only consider contracts with available rates
    available = [c for c in contracts_sorted if pd.notna(row[c])]

    if len(available) < 2:
        # Not enough contracts, cannot hold 2nd
        held_contract.append(None)
        continue

    front, second = available[0], available[1]

    if current_contract is None:
        # Start by holding the 2nd contract
        current_contract = second
    elif current_contract == front:
        # Roll ONLY when our held contract becomes front
        current_contract = second
    # else: continue holding current_contract

    held_contract.append(current_contract)

sofr_rates['held_contract'] = held_contract

# --- Daily held rate ---
sofr_rates['held_rate'] = [
    sofr_rates.loc[d, c] if c is not None else np.nan
    for d, c in zip(sofr_rates.index, sofr_rates['held_contract'])
]

# --- P&L calculation ---
sofr_rates['delta_r'] = sofr_rates['held_rate'].diff()
sofr_rates['pnl'] = -sofr_rates['delta_r'] * 100 * 25  # DV01 = $25/bp
sofr_rates['pnl'].iloc[0] = 0
sofr_rates['cum_pnl'] = sofr_rates['pnl'].cumsum()

# --- Performance metrics ---
daily_pnl = sofr_rates['pnl'].dropna()
mean_daily_pnl = daily_pnl.mean()
std_daily_pnl = daily_pnl.std()
sharpe_ratio = np.sqrt(252) * mean_daily_pnl / std_daily_pnl

sofr_rates['running_max'] = sofr_rates['cum_pnl'].cummax()
sofr_rates['drawdown'] = sofr_rates['cum_pnl'] - sofr_rates['running_max']
max_drawdown = sofr_rates['drawdown'].min()

# --- Enhancement: only hold when curve is upward sloping (2nd >= front) ---
enh_pnl = []
enh_held_contract = []
prev_rate = None
prev_contract = None

for date, row in sofr_rates.iterrows():
    available = [c for c in contracts_sorted if pd.notna(row[c])]
    if len(available) < 2:
        enh_pnl.append(np.nan)
        enh_held_contract.append(None)
        prev_rate = None
        prev_contract = None
        continue

    front, second = available[0], available[1]
    r1, r2 = row[front], row[second]

    # Signal: hold only if curve is upward sloping
    hold = r2 >= r1
    if not hold:
        enh_pnl.append(0.0)
        enh_held_contract.append(None)
        prev_rate = None
        prev_contract = None
        continue

    if prev_rate is None or prev_contract != second:
        pnl = 0.0
    else:
        pnl = -(r2 - prev_rate) * 100 * 25

    enh_pnl.append(pnl)
    enh_held_contract.append(second)
    prev_rate = r2
    prev_contract = second

sofr_rates['enh_pnl'] = enh_pnl
sofr_rates['enh_cum_pnl'] = pd.Series(enh_pnl, index=sofr_rates.index).fillna(0).cumsum()

enh_daily = sofr_rates['enh_pnl'].dropna()
enh_mean = enh_daily.mean()
enh_std = enh_daily.std()
enh_sharpe = np.sqrt(252) * enh_mean / enh_std if enh_std != 0 else np.nan

enh_running_max = sofr_rates['enh_cum_pnl'].cummax()
enh_drawdown = sofr_rates['enh_cum_pnl'] - enh_running_max
enh_max_drawdown = enh_drawdown.min()

# --- Print results ---
print("="*60)
print("SOFR 2ND CONTRACT CARRY STRATEGY")
print("="*60)
print(f"Backtest Period: {sofr_rates.index[0].date()} → {sofr_rates.index[-1].date()}")
print(f"Total Trading Days: {len(sofr_rates)}")
print(f"\nCumulative P&L:      ${sofr_rates['cum_pnl'].iloc[-1]:,.2f}")
print(f"Sharpe Ratio:        {sharpe_ratio:.3f}")
print(f"Maximum Drawdown:    ${max_drawdown:,.2f}")
print(f"\nAverage Daily P&L:   ${mean_daily_pnl:,.2f}")
print(f"P&L Std Dev:         ${std_daily_pnl:,.2f}")
print("="*60)

print("\n" + "="*60)
print("ENHANCED STRATEGY (HOLD ONLY IF 2ND >= FRONT)")
print("="*60)
print(f"Cumulative P&L:      ${sofr_rates['enh_cum_pnl'].iloc[-1]:,.2f}")
print(f"Sharpe Ratio:        {enh_sharpe:.3f}")
print(f"Maximum Drawdown:    ${enh_max_drawdown:,.2f}")
print(f"\nAverage Daily P&L:   ${enh_mean:,.2f}")
print(f"P&L Std Dev:         ${enh_std:,.2f}")
print("="*60)
