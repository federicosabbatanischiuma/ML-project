#!/usr/bin/env python
# coding: utf-8

"""
Portfolio Optimization with Transformer Neural Networks
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functions_file import MultiHeadAttention, FeedForward, TransformerBlock, NonlinearPortfolioForward, set_seed

def main():
    """Main function to run the portfolio optimization experiment."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration - Change these variables to test different models
    size_group = 'micro'  # Options: 'micro', 'small', 'mid', 'mega'
    constraint_type = 'vanilla'  # Options: 'vanilla', 'short_constraint', 'leverage_constraint'
    
    # Output filenames
    csv_filename = f"{size_group}_{constraint_type}_portfolio_results.csv"
    plot_filename = f"{size_group}_{constraint_type}_portfolio_plot.png"

    # Data loading
    dataset_path = "../Data/our_version_norm.pkl"

    stock_data = pd.read_pickle(dataset_path)
    stock_data = stock_data[stock_data["size_grp"] == size_group]

    # Hyperparameters initialization

    months_list = stock_data["date"].unique()
    columns_to_drop_in_x = ["size_grp", "date", "r_1", "id"]
    window = 60
    epoch = 20
    K = 10
    D = stock_data.shape[1] - len(columns_to_drop_in_x)
    H = 1
    dF = 256
    lr = 1e-4
    
    if constraint_type == 'vanilla':
        ridge_penalty = 10
    elif constraint_type == 'short_constraint':
        ridge_penalty = 0.001
    else:  # leverage_constraint
        ridge_penalty = 0.01
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training loop
    portfolio_ret = []
    dates_to_save = []
    first_t = 61
    last_T = len(months_list) - 1
    
    for t in range(first_t, last_T):
        print(t)
        model = NonlinearPortfolioForward(
            D=D, K=K, H=H, dF=dF, constraint_type=constraint_type
        ).to(device)
        
        if constraint_type == 'leverage_constraint':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) #we detected better results adding a weight decay
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for e in range(epoch):
            for month in months_list[t - window:t]:
                month_data = stock_data[stock_data["date"] == month]
                
                X_t = month_data.drop(columns=columns_to_drop_in_x)
                R_t_plus_one = torch.tensor(
                    month_data["r_1"].values,
                    dtype=torch.float32,
                    device=device
                )
                
                X_t_tensor = torch.tensor(X_t.values, dtype=torch.float32, device=device)
                w_t = model(X_t_tensor)
                
                loss = (1 - torch.dot(w_t, R_t_plus_one)).pow(2) + ridge_penalty * torch.norm(w_t, p=2).pow(2)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Out-of-sample prediction
        month_data = stock_data[stock_data["date"] == months_list[t]]
        X_t = month_data.drop(columns=columns_to_drop_in_x)
        R_t_plus_one = torch.tensor(
            month_data["r_1"].values,
            dtype=torch.float32,
            device=device
        )
        
        X_t_tensor = torch.tensor(X_t.values, dtype=torch.float32, device=device)
        w_t = model(X_t_tensor)
        
        predicted = (w_t @ R_t_plus_one).item()
        portfolio_ret.append(predicted)
        dates_to_save.append(months_list[t+1])

    # Save results
    data = {
        "Date": dates_to_save,
        "Return": portfolio_ret,
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)
    
    # Plot results
    portfolio_cum_return = np.cumsum(np.asarray(portfolio_ret))
    
    # Calculate Sharpe Ratio
    ret = np.array(portfolio_ret)
    mean = ret.mean()
    std = ret.std(ddof=1)
    sharpe_ratio = np.sqrt(12) * mean / std
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates_to_save, portfolio_cum_return, label=f"{constraint_type.replace('_', ' ').title()} Portfolio")

    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.title(f"Cumulative Returns - {constraint_type.replace('_', ' ').title()}: "
              f"epochs={epoch}, H={H}, K={K}, z={ridge_penalty}, SR={sharpe_ratio:.2f}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plot_path = plot_filename
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    main()
