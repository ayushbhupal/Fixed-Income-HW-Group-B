import pandas as pd
import numpy as np

# --- Identify SOFR contracts ---
contracts = [c for c in sofr_rates.columns if c.startswith("SR3")]
month_map = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}

def parse_contract(c):
    month = month_map[c[3]]
    year = 2020 + int(c[4])
    return (year, month)

contracts_sorted = sorted(contracts, key=parse_contract)
sofr_rates = sofr_rates[contracts_sorted].copy()

# --- Strategy 1: Long 2nd, Short Front only if front > 2nd ---
held_contract = []
front_contract = []
current_contract = None

for date, row in sofr_rates.iterrows():
    available = [c for c in contracts_sorted if pd.notna(row[c])]
    if len(available) < 2:
        held_contract.append(None)
        front_contract.append(None)
        continue

    front, second = available[0], available[1]

    # --- Only open position if front rate > second rate ---
    if row[front] > row[second]:
        if current_contract is None or current_contract == front:
            current_contract = second
    else:
        current_contract = None  # flat if condition not met

    held_contract.append(current_contract)
    front_contract.append(front if current_contract is not None else None)

sofr_rates['held_contract'] = held_contract
sofr_rates['front_contract'] = front_contract

# --- Daily rates ---
sofr_rates['held_rate'] = [
    sofr_rates.loc[d, c] if c is not None else np.nan
    for d, c in zip(sofr_rates.index, sofr_rates['held_contract'])
]
sofr_rates['front_rate'] = [
    sofr_rates.loc[d, c] if c is not None else np.nan
    for d, c in zip(sofr_rates.index, sofr_rates['front_contract'])
]

# --- P&L ---
sofr_rates['delta_held'] = sofr_rates['held_rate'].diff()
sofr_rates['delta_front'] = sofr_rates['front_rate'].diff()

# Handle days with no position
sofr_rates['delta_held'] = sofr_rates['delta_held'].fillna(0)
sofr_rates['delta_front'] = sofr_rates['delta_front'].fillna(0)

# P&L: long 2nd, short front
sofr_rates['pnl'] = -sofr_rates['delta_held'] * 100 * 25 + sofr_rates['delta_front'] * 100 * 25
sofr_rates['pnl'].iloc[0] = 0
sofr_rates['cum_pnl'] = sofr_rates['pnl'].cumsum()

# --- Metrics ---
daily_pnl = sofr_rates['pnl']
mean_daily_pnl = daily_pnl.mean()
std_daily_pnl = daily_pnl.std()
sharpe_ratio = np.sqrt(252) * mean_daily_pnl / std_daily_pnl

sofr_rates['running_max'] = sofr_rates['cum_pnl'].cummax()
sofr_rates['drawdown'] = sofr_rates['cum_pnl'] - sofr_rates['running_max']
max_drawdown = sofr_rates['drawdown'].min()

print("="*60)
print("STRATEGY 1: LONG 2ND, SHORT FRONT (front > 2nd)")
print("="*60)
print(f"Cumulative P&L:      ${sofr_rates['cum_pnl'].iloc[-1]:,.2f}")
print(f"Sharpe Ratio:        {sharpe_ratio:.3f}")
print(f"Maximum Drawdown:    ${max_drawdown:,.2f}")
print("="*60)

# plot cumulative P&L
fig, ax = plt.subplots(figsize=figsize)
ax.plot(sofr_rates.index, sofr_rates['cum_pnl'], color='blue', label='Cumulative P&L')
ax.set_title('Cumulative P&L - Strategy 1')
ax.set_ylabel('P&L ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
