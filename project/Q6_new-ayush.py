import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Identify & sort SOFR contracts
# ===============================
contracts = [c for c in sofr_rates.columns if c.startswith("SR3")]

month_map = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}

def parse_contract(c):
    month = month_map[c[3]]
    year = 2020 + int(c[4])
    return (year, month)

contracts_sorted = sorted(contracts, key=parse_contract)
sofr_rates = sofr_rates[contracts_sorted].copy()

# ===============================
# Strategy: Long 2nd, Short Front
# Only if front > second
# ===============================
held_contract = []
front_contract = []
position = []

current_held = None
current_front = None
in_position = False

for date, row in sofr_rates.iterrows():
    available = [c for c in contracts_sorted if pd.notna(row[c])]

    if len(available) < 2:
        held_contract.append(None)
        front_contract.append(None)
        position.append(0)
        in_position = False
        continue

    front, second = available[0], available[1]

    if row[front] > row[second]:
        # open or continue position
        current_held = second
        current_front = front
        in_position = True
    else:
        # flat
        current_held = None
        current_front = None
        in_position = False

    held_contract.append(current_held)
    front_contract.append(current_front)
    position.append(int(in_position))

sofr_rates['held_contract'] = held_contract
sofr_rates['front_contract'] = front_contract
sofr_rates['position'] = position

# ===============================
# Fetch daily rates
# ===============================
sofr_rates['held_rate'] = [
    sofr_rates.loc[d, c] if c is not None else np.nan
    for d, c in zip(sofr_rates.index, sofr_rates['held_contract'])
]

sofr_rates['front_rate'] = [
    sofr_rates.loc[d, c] if c is not None else np.nan
    for d, c in zip(sofr_rates.index, sofr_rates['front_contract'])
]

# ===============================
# Daily rate changes
# ===============================
sofr_rates['delta_held'] = sofr_rates['held_rate'].diff()
sofr_rates['delta_front'] = sofr_rates['front_rate'].diff()

# ===============================
# Kill P&L on:
# - entry
# - exit
# - contract change (roll)
# ===============================
contract_change = (
    (sofr_rates['held_contract'] != sofr_rates['held_contract'].shift()) |
    (sofr_rates['front_contract'] != sofr_rates['front_contract'].shift()) |
    (sofr_rates['position'] != sofr_rates['position'].shift())
)

sofr_rates.loc[contract_change, ['delta_held', 'delta_front']] = 0.0

sofr_rates[['delta_held', 'delta_front']] = sofr_rates[
    ['delta_held', 'delta_front']
].fillna(0)

# ===============================
# P&L calculation
# Long 2nd, Short Front
# ===============================
DV01 = 100 * 25  # SOFR contract scaling

sofr_rates['pnl'] = (
    -sofr_rates['delta_held'] * DV01 +
     sofr_rates['delta_front'] * DV01
)

sofr_rates['cum_pnl'] = sofr_rates['pnl'].cumsum()

# ===============================
# Performance metrics
# ===============================
active_pnl = sofr_rates.loc[sofr_rates['position'] == 1, 'pnl']

mean_daily_pnl = active_pnl.mean()
std_daily_pnl = active_pnl.std()

sharpe_ratio = (
    np.sqrt(252) * mean_daily_pnl / std_daily_pnl
    if std_daily_pnl > 0 else np.nan
)

sofr_rates['running_max'] = sofr_rates['cum_pnl'].cummax()
sofr_rates['drawdown'] = sofr_rates['cum_pnl'] - sofr_rates['running_max']
max_drawdown = sofr_rates['drawdown'].min()

# ===============================
# Output
# ===============================
print("=" * 60)
print("STRATEGY: LONG 2ND, SHORT FRONT (front > 2nd)")
print("=" * 60)
print(f"Cumulative P&L:      ${sofr_rates['cum_pnl'].iloc[-1]:,.2f}")
print(f"Sharpe Ratio:        {sharpe_ratio:.3f}")
print(f"Maximum Drawdown:    ${max_drawdown:,.2f}")
print("=" * 60)

# ===============================
# Plot
# ===============================
plt.figure(figsize=(12, 5))
plt.plot(sofr_rates.index, sofr_rates['cum_pnl'], label='Cumulative P&L')
plt.title('Cumulative P&L – SOFR Curve Inversion Strategy')
plt.ylabel('P&L ($)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
