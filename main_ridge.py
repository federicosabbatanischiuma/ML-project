"""main_ridge.py
============================================
End‑to‑end driver script for the **random‑feature ridge‑regression back‑test**.

This program does three main things:

1. **Load data**  – reads ``our_version_norm.pkl`` (stacked asset returns and
   signal variables) from a *data* folder one level above the script’s
   directory.
2. **Random‑feature experiment**  – for a grid of feature counts *P* it
   generates sine–cosine random‑kitchen‑sink features, builds managed returns,
   runs an ridge portfolio and stores
   Sharpe ratios / cumulative returns.

Outputs (under ``Results/Results_for_<size>``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``sharpe_ratios_by_P_<size>.csv``  – Sharpe vs complexity grid
* ``cumulative_returns_by_P_<size>.csv``  – cum‑returns per (P, λ)
* ``all_cumrets_<size>.png``  – cumulative‑return plot
* ``sharpe_vs_P_<size>.png``  – Sharpe vs P plot

Adjust the **`size_group`** variable (mega/large/small/micro) to focus on a
particular market‑cap bucket, and tweak ``complexities`` or shrinkage values as
needed.  All heavy lifting lives in *functions_ridge.py*.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from functions_ridge import (
    build_managed_returns,
    produce_random_feature_managed_returns,
    hw_efficient_portfolio_oos,
    regression_with_alpha_and_tstat,
    hw_efficient_portfolio_oos_2
)

# Load data
dataset_path = "../Data/our_version_norm.pkl"
stock_data = pd.read_pickle(dataset_path)

# HERE, SELECT the desired SIZE GROUP from : 'mega', 'large', 'small', 'micro'
size_group = 'mega'
if size_group is not None:
  stock_data = stock_data.loc[stock_data.size_grp==size_group]

# Set output directory
output_dir = Path(__file__).resolve().parent/ "Results"/ f"Results_for_{size_group}"
output_dir.mkdir(parents=True, exist_ok=True)

# Obtain  managed returns and signals
stock_data.set_index(["id", "date"], inplace=True)
size_groups = stock_data.pop('size_grp')
signals = stock_data.drop(columns=['r_1'])
hw_managed_returns = build_managed_returns(returns = stock_data['r_1'], signals = signals)


# Define complexity levels
complexities = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
sharpe_by_P = {}     # {P: {z: Sharpe}}
returns_by_P = {}    # {(P, z): return_series}

# Loop over complexity levels
for P in complexities:
    print(f"Processing P = {P}...")
    hw_random_feature_managed_returns = produce_random_feature_managed_returns(P, stock_data, signals)
    oos_df, sharpe_dict = hw_efficient_portfolio_oos(hw_random_feature_managed_returns,P)

    sharpe_by_P[P] = sharpe_dict

    for shrink, series in oos_df.items():
        standardized = series / series.std()
        returns_by_P[(P, shrink)] = standardized

# Save Sharpe ratios to CSV
sharpe_df = pd.DataFrame.from_dict(sharpe_by_P, orient='index')  # rows = P, cols = z
sharpe_df.index.name = "P"
sharpe_df.to_csv(output_dir / f"sharpe_ratios_by_P_{size_group}.csv")
print("Saved Sharpe ratios to CSV.")

# Save cumulative returns to CSV
returns_df = pd.DataFrame({
    f"P={P}_z={shrink}": ret.cumsum()
    for (P, shrink), ret in returns_by_P.items()
})
returns_df.index.name = "Date"
returns_df.to_csv(output_dir / f"cumulative_returns_by_P_{size_group}.csv")
print("Saved cumulative returns to CSV.")

# Plot cumulative returns
plt.figure(figsize=(12, 6))
for label, cumret in returns_df.items():
    cumret.plot(label=label)

plt.title("Cumulative Returns by Complexity P and Shrinkage z")
plt.xlabel("Date")
plt.ylabel("Cumulative Standardized Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / f"all_cumrets_{size_group}.png")
plt.close()
print("Saved cumulative returns plot.")

# Plot Sharpe ratios vs P for each shrinkage level
plt.figure(figsize=(8, 5))
for shrink in sharpe_df.columns:
    plt.plot(sharpe_df.index, sharpe_df[shrink], marker='o', label=f"z={shrink}")

plt.title("Sharpe Ratio vs Complexity P")
plt.xlabel("Number of Random Features (P)")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / f"sharpe_vs_P_{size_group}.png")
plt.close()
print("Saved Sharpe ratio plot.")

