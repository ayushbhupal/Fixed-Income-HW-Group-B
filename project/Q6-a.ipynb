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
