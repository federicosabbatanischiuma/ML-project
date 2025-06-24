import pandas as pd

# -------------------------------------------------------
# Configuration
# Set the returns_name to match the output from main.py
returns_name = "vanilla_portfolio_results.csv"  # Change this based on your main.py output
                                                # Options: "vanilla_portfolio_results.csv"
                                                #         "short_constraint_results.csv" 
                                                #         "leverage_constraint_results.csv"

window_periods = 12  # Number of periods for rolling volatility window
# -------------------------------------------------------

# Read the CSV from main.py output
df = pd.read_csv(returns_name, parse_dates=['Date'], index_col='Date')

# Calculate rolling volatility (standard deviation of returns)
rolling_vol = df['Return'].rolling(window=window_periods).std()

# Normalize returns by rolling volatility
df['Normalized_Return'] = df['Return'] / rolling_vol

# Save the result
output_name = f"vol_normalized_{returns_name}"
df.to_csv(output_name)

print(f"Volatility-normalized returns saved to: {output_name}")
