from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#If you want to plot both raw and volatility managed data uncomment the commented lines below
out_dir = Path("project_results_constrained")  

raw_path     = "lele_DF_large.csv"
#managed_path = "3normalized_lele_DF_large.csv.csv"

raw     = pd.read_csv(raw_path,     index_col=0, parse_dates=True)
#managed = pd.read_csv(managed_path, index_col=0, parse_dates=True)
#common_dates = raw.index.intersection(managed.index)

# raw     = raw.loc[common_dates].copy()
# managed = managed.loc[common_dates].copy()
cum_raw = raw["Return"].cumsum()
# cum_man = managed["Return"].cumsum()

def sharpe(series):
    return (series.mean() / series.std(ddof=0)) * np.sqrt(12)

print("Sharpe annualized      :", sharpe(raw["Return"]).round(3))
#print("Sharpe annualizzato (Vol-managed):", sharpe(managed["Return"]).round(3))

# ---- 4. plot ----
plt.figure(figsize=(8,6))
plt.plot(cum_raw, label="Cumsum  lev_constraint")
# plt.plot(cum_man, label="Vol-managed cumsum")
# plt.title(f"Cumulative SUM of returns . SR = {sharpe(raw['Return']).round(3)} (Raw) SR = {sharpe(managed['Return']).round(3)} (Vol-managed)")
plt.title(f"Leverage constraint with z=0.1. SR = {sharpe(raw['Return']).round(3)} ") #comment this line if you want to plot the managed data as well
plt.ylabel("Cumulative returns")
plt.legend(); 
plt.grid(True)
plt.tight_layout()

plot_file =  f"{raw_path}_cumulative.png"
# plot_file =  f"{managed_path}_cumulative.png"
plt.savefig(plot_file, dpi=150)
plt.close()
