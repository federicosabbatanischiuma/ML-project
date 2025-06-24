
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from functions_ridge import (
    build_managed_returns,
    produce_random_feature_managed_returns,
    sharpe_ratio,
    regression_with_alpha_and_tstat,
    hw_efficient_portfolio_oos_2
)

# %%
stock_data = pd.read_pickle("../Data/our_version_norm.pkl")
stock_data.set_index(["id", "date"], inplace=True)
size_group = 'micro'
if size_group is not None:
  stock_data = stock_data.loc[stock_data.size_grp==size_group]





ff6_columns = ['ret_1_0', 'market_equity', 'be_me', 'op_at', 'at_gr1', 'ret_12_1']
sy_columns = ['ret_1_0', 'mispricing_mgmt', 'mispricing_perf']
hxz_columns = ['ret_1_0', 'market_equity', 'at_gr1', 'op_at', 'niq_at_chg1']  # or use 'niq_be_chg1' or 'saleq_su' for EG
dhs_columns = ['ret_1_0', 'ret_12_1']  # 'ret_12_1' used for both MOM and LT reversal (inverted)
bs_columns = ['ret_1_0', 'at_gr1', 'op_at', 'market_equity', 'ret_12_1', 'be_me']

models = ['FF6', 'SY', 'HXZ', 'DHS', 'BS']

columns_list = [ff6_columns, sy_columns, hxz_columns, dhs_columns, bs_columns]

# %%
##benchamrks first
rets_list = []
for model, name in zip(columns_list, models):
    rets = build_managed_returns(stock_data['r_1'], stock_data[model])
    df = hw_efficient_portfolio_oos_2(rets, name)
    df.columns = [name]
    rets_list.append(df)

all_returns = pd.concat(rets_list, axis=1)


# %%

# Load data
my_model_ret_df = pd.read_csv("..\Data\cumulative_returns_by_P_micro.csv")
my_model_ret_df = my_model_ret_df[["Date", 'P=300_λ=1000']] ## pick the right column if the file has more than one
its_cumulative = True

# If cumulative, convert to non-cumulative
if its_cumulative:
    date_column = my_model_ret_df.iloc[:, 0]
    cumulative_data = my_model_ret_df.iloc[:, 1:]
    true_values = np.vstack([
        cumulative_data.iloc[0].values, 
        np.diff(cumulative_data.values, axis=0)
    ])
    
    # Construct DataFrame from true (non-cumulative) values
    true_df = pd.DataFrame(true_values, columns=cumulative_data.columns)
    true_df.insert(0, "Date", date_column)

    my_model_ret_df = true_df

# Rename column and set index
model_name = "Ridge"
my_model_ret_df = my_model_ret_df[["Date", 'P=300_λ=1000']]
my_model_ret_df.columns = ["Date", model_name]
my_model_ret_df.set_index("Date", inplace=True)

# Ensure datetime index
my_model_ret_df.index = pd.to_datetime(my_model_ret_df.index)
all_returns.index = pd.to_datetime(all_returns.index)

# Final filtered result
my_model_ret_df = my_model_ret_df[my_model_ret_df.index >= all_returns.index[0]]
my_model_ret_df

# %%
all_returns = pd.concat([my_model_ret_df, all_returns], axis=1, join='inner')
all_returns

# %%
cumulative = all_returns.cumsum()

# Compute Sharpe ratios
sharpe_ratios = {col: sharpe_ratio(all_returns[col]) for col in all_returns.columns}
print(sharpe_ratios)
# Plot
fig, ax = plt.subplots(figsize=(10,6))
cumulative.plot(ax=ax)

# Custom legend with Sharpe ratios
handles, _ = ax.get_legend_handles_labels()
new_labels = [f"{col} (SR={sharpe_ratios[col]})" for col in all_returns.columns]
ax.legend(handles, new_labels)

# Final touches
plt.title(f'Size: {size_group}')
plt.ylabel('Cumulative Return')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_returns_by_model_transf.png", dpi=300)
plt.show()

# %%

# Assign colors: red for the max, blue for the rest
colors = ['red' if i ==0 else 'blue' for i in range(7)]


# Plot
plt.figure(figsize=(5.5, 6))
plt.bar(sharpe_ratios.keys(), sharpe_ratios.values(), color=colors)

# Add labels and title
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratios of Strategies')
plt.xticks(rotation=30, ha='right')  # Rotate x-axis labels

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show
plt.savefig('sharpe_ratios_barplot_transf.png', dpi=300)
plt.show()

# %%
import statsmodels.api as sm

def regression_with_alpha_and_tstat(predicted_variable, explanatory_variables):
    x_ = sm.add_constant(explanatory_variables)
    y_ = predicted_variable
    z_ = x_.copy().astype(float)
    result = sm.OLS(y_.values, z_.values).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    
    alpha = result.params[0]         # intercept (alpha)
    alpha_tstat = result.tvalues[0]  # t-statistic of intercept

    return alpha, alpha_tstat

v = {}
for col in all_returns.columns:
    v[col]=regression_with_alpha_and_tstat(my_model_ret_df, all_returns[col])

print(pd.DataFrame(v)
)

