#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

def set_seed(seed=42):
    """Set seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for transformer."""
    
    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.W = nn.ParameterList([nn.Parameter(torch.randn(D, D)/100) for _ in range(H)])
        self.V = nn.ParameterList([nn.Parameter(torch.randn(D, D)/100) for _ in range(H)])

    def forward(self, X):  # X: [N_t, D]
        heads = []
        for h in range(self.H):
            scores = X @ self.W[h] @ X.T / (X.shape[1] ** 0.5)  # [N_t, N_t]
            weights = F.softmax(scores, dim=1) + 1e-8  # softmax row-wise
            A_h = weights @ X @ self.V[h]  # [N_t, D]
            heads.append(A_h)
        return sum(heads)  # [N_t, D]

class FeedForward(nn.Module):
    """Feed-forward network for transformer."""
    
    def __init__(self, D, dF):
        super().__init__()
        self.fc1 = nn.Linear(D, dF)
        self.fc2 = nn.Linear(dF, D)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):  # X: [N_t, D]
        return self.dropout(self.fc2(F.relu(self.fc1(X))))  # [N_t, D]

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, D, H, dF):
        super().__init__()
        self.attn = MultiHeadAttention(D, H)
        self.ffn = FeedForward(D, dF)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, X):  # X: [N_t, D]
        X = self.norm1(X + self.attn(X))  # normalize after attention residual
        X = self.norm2(X + self.ffn(X))   # normalize after FFN residual
        return X

class NonlinearPortfolioForward(nn.Module):
    """
    Nonlinear Portfolio Forward model with different constraint variants.
    
    Args:
        D: Feature dimension
        K: Number of transformer blocks
        H: Number of attention heads
        dF: Feed-forward dimension
        constraint_type: 'vanilla', 'short_constraint', or 'leverage_constraint'
    """
    
    def __init__(self, D, K, H=1, dF=256, constraint_type='vanilla'):
        super().__init__()
        self.constraint_type = constraint_type
        self.blocks = nn.ModuleList([TransformerBlock(D, H, dF) for _ in range(K)])
        self.lambda_out = nn.Parameter(torch.randn(D, 1)/1000)  # final projection
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        for block in self.blocks:
            X = block(X)
        
        w_t = X @ self.lambda_out.squeeze()
        
        if self.constraint_type == 'vanilla':
            return w_t
        elif self.constraint_type == 'short_constraint':
            w_t = F.relu(w_t)
            return w_t
        elif self.constraint_type == 'leverage_constraint':
            leverage = w_t.abs().sum() + 1e-6  # avoid div by zero
            scaling = torch.clamp(1.5 / leverage, max=1.0)
            w = w_t * scaling
            return w
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")



def Subsample_OptimalRidge(constraint_type, stock_data, months_list, window, epochs, K, H, dF, lr, ridge_penalties):
    """
    Function to train and evaluate the NonlinearPortfolioForward model with different ridge penalties.
    Args:
        constraint_type: Type of constraint ('vanilla', 'short_constraint', 'leverage_constraint')
        stock_data: DataFrame containing stock data
        months_list: List of unique months in the dataset
        window: Size of the training window
        epochs: Number of training epochs
        K: Number of transformer blocks
        H: Number of attention heads
        dF: Feed-forward dimension
        lr: Learning rate
        ridge_penalties: List of ridge penalties to evaluate
    Returns:
        None: Saves the cumulative returns plot for each ridge penalty.
    """
    #!/usr/bin/env python
    # coding: utf-8

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Directory di output
    output_dir = Path("CV_folder")
    output_dir.mkdir(exist_ok=True)

    dataset_path = "data/our_version_norm.pkl"
    benchmark_path = "data/SandP benchmark.csv"

    stock_data = pd.read_pickle(dataset_path)
    #subsample the dataset
    stock_data = stock_data[stock_data["size_grp"] == "micro"]
    stock_data = stock_data[stock_data["id"] % 6  == 0]
    stock_data = stock_data[stock_data["id"] % 5  == 0]

    SP_benchmark = pd.read_csv(benchmark_path)
    SP_benchmark["caldt"]         = pd.to_datetime(SP_benchmark["caldt"])
    SP_benchmark["caldt_period"]  = SP_benchmark["caldt"].dt.to_period("M")

    months_list = stock_data["date"].sort_values().unique()
    columns_to_drop = ["size_grp", "date", "r_1", "id"]

    window         = 60
    epochs         = 20
    K              = 10
    H              = 1
    dF             = 256
    lr             = 1e-4
    if constraint_type == 'vanilla':
        ridge_penalties = np.logspace(-2, 3, 6)
    else:  # short_constraint or leverage_constraint
        ridge_penalties = np.logspace(-4, 1, 6)

    first_t = window + 1
    last_t  = len(months_list) - 1

    plt.figure(figsize=(12, 6))
    for ridge in ridge_penalties:
        rets_list = []

        for t in range(first_t, last_t):
            D = stock_data.shape[1] - len(columns_to_drop)
            model = NonlinearPortfolioForward(D=D, K=K, H=H, dF=dF, constraint_type=constraint_type).to(device)
            optim_ = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

            for ep in range(epochs):
                for month in months_list[t-window : t]:
                    md     = stock_data[stock_data["date"] == month]
                    X_t    = torch.tensor(md.drop(columns=columns_to_drop).values,
                                        dtype=torch.float32, device=device)
                    R_next = torch.tensor(md["r_1"].values,
                                        dtype=torch.float32, device=device)
                    w_t    = model(X_t)
                    loss   = (1 - torch.dot(w_t, R_next)).pow(2) \
                            + ridge * torch.norm(w_t, p=2).pow(2)
                    optim_.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim_.step()

            with torch.no_grad():
                md      = stock_data[stock_data["date"] == months_list[t]]
                X_t     = torch.tensor(md.drop(columns=columns_to_drop).values,
                                    dtype=torch.float32, device=device)
                R_next  = torch.tensor(md["r_1"].values,
                                    dtype=torch.float32, device=device)
                w_next  = model(X_t)
                ret_t   = (w_next @ R_next).item()
                rets_list.append(ret_t)
                
        SR = rets_list.mean()/rets_list.std() *np.sqrt(12)
        cum_rets = np.cumsum(rets_list)
        plt.plot(months_list[first_t:last_t], cum_rets,
                label=f"ridge={ridge:.0e}, SR ={SR:.0e}")

    dates_period = pd.Series(months_list[first_t:last_t]).astype("datetime64[ns]").tolist()
    sp = SP_benchmark[SP_benchmark["caldt_period"]
                    .isin(pd.Series(months_list[first_t:last_t]).astype("period[M]"))]
    sp = sp.sort_values("caldt")
    sp_cum = np.cumsum(sp["vwretd"].values)
    plt.plot(sp["caldt"], sp_cum, "--", color="k", label="S&P 500")

    plt.title("Constrained Portfolio with lr=1e-4, GridSearch over ridge penalties")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()

    plt.savefig(output_dir / "cumsum_all_ridge.png")
    plt.close()

    print("Plot saved in:", output_dir / "cumsum_all_ridge.png")