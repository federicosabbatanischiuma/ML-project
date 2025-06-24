"""functions_ridge.py
--------------------------------------------------
Helper functions for the ridge‑regularised random‑feature portfolio experiments used by 'main_ridge.py'.

Contents:

- :func:`build_managed_returns` – collapse panel returns into managed returns
- :func:`sharpe_ratio` – monthly‑data Sharpe (annualised)
- :func:`ridge_regr` – vectorised closed‑form ridge, many z at once
- :func:`hw_efficient_portfolio_oos` – Horvitz–Wermuth out‑of‑sample back‑test
- :func:`produce_random_feature_managed_returns` – random Fourier factors
- :func:`regression_with_alpha_and_tstat` – HAC‑robust α & t‑stat
- :func:`hw_efficient_portfolio_oos_2` – variant with fixed z list

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_managed_returns(returns, signals):
  """Return **managed returns** time‑series.

    Parameters
    ----------
    returns : pd.Series
        Stacked panel of asset returns, indexed by *(id, date)*.
    signals : pd.DataFrame
        Stacked panel of signals with the same index as *returns*.  Each column
        is a factor/feature; each row an *(id, date)* observation.

    Returns
    -------
    pd.DataFrame
        Date‑indexed DataFrame whose columns correspond to the original signal
        columns and whose rows are daily/monthly total managed returns, i.e.
        ``sum_i signal_{i,t} * return_{i,t}``.
    """
  managed_returns = (signals * returns.values.reshape(-1, 1)).groupby(signals.index.get_level_values('date')).sum()
  return managed_returns

def sharpe_ratio(returns):
  """Annualised Sharpe ratio for **monthly** return data.

    Notes
    -----
    The factor ``sqrt(12)`` annualises a monthly series.  Results are rounded
    to two decimal places purely for display; remove ``np.round`` if you need
    full precision downstream.
    """
  return np.round(np.sqrt(12) * returns.mean() / returns.std(), 2)

def ridge_regr(signals: np.ndarray,
                  labels: np.ndarray,
                  future_signals: np.ndarray,
                  shrinkage_list: np.ndarray):
    """Closed‑form ridge for many z values

    Regression is
    beta = (zI + S'S/t)^{-1}S'y/t = S' (zI+SS'/t)^{-1}y/t
    Inverting matrices is costly, so we use eigenvalue decomposition:
    (zI+A)^{-1} = U (zI+D)^{-1} U' where UDU' = A is eigenvalue decomposition,
    and we use the fact that D @ B = (diag(D) * B) for diagonal D, which saves a lot of compute cost.

    Parameters
    ----------
    signals : ndarray of shape *(t, p)*
        In‑sample signal matrix *S*.
    labels : ndarray of shape *(t, 1)*
        Target vector *y* (ones if replicating an equal‑weight portfolio).
    future_signals : ndarray of shape *(1, p)*
        Out‑of‑sample signal row for which we want a prediction.
    shrinkage_list : 1‑D array‑like
        List/array of ridge parameters ``z``.

    Returns
    -------
    betas : ndarray of shape *(p, len(shrinkage_list))*
        Coefficients for each z.
    predictions : ndarray of shape *(len(shrinkage_list),)*
        OOS predictions for each z.
    """
    t_ = signals.shape[0]
    p_ = signals.shape[1]
    if p_ < t_:
        # this is standard regression
        eigenvalues, eigenvectors = np.linalg.eigh(signals.T @ signals / t_)
        means = signals.T @ labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)
        betas = eigenvectors @ intermed
    else:
        # this is the weird over-parametrized regime
        eigenvalues, eigenvectors = np.linalg.eigh(signals @ signals.T / t_)
        means = labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means # this is \mu

        # now we build [(z_1+\delta)^{-1}, \cdots, (z_K+\delta)^{-1}] * \mu
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)

        tmp = eigenvectors.T @ signals # U.T @ S
        betas = tmp.T @ intermed # (S.T @ U) @ [(z_1+\delta)^{-1}, \cdots, (z_K+\delta)^{-1}] * \mu
    predictions = future_signals @ betas
    return betas, predictions

def hw_efficient_portfolio_oos(raw_factor_returns: pd.DataFrame, P: int, shrinkage_list=[1, 10, 100, 1000, 10000,1000000]):
    """
    At each step *t*:
      1. Fit ridge to 360 months of history.
      2. Predict the one‑period weight vector.
      3. Compute and store the realised return.

    Parameters
    ----------
    raw_factor_returns : DataFrame (T × P)
        Each column is a factor return series.
    P : int
        Only used for console printouts – number of factors/random features.
    shrinkage_list : list[int]
        z values to sweep.
    """
    oos_returns = []
    dates = []

    for t in range(360, len(raw_factor_returns)):
        X_train = raw_factor_returns.iloc[t-360:t, :].values
        y_train = np.ones((X_train.shape[0], 1)) 
        X_test = raw_factor_returns.iloc[t, :].values

        beta, optimal = ridge_regr(signals=X_train,
                                   labels=y_train,
                                   future_signals=X_test,
                                   shrinkage_list=shrinkage_list)
        oos_returns.append(optimal)
        dates.append(raw_factor_returns.index[t])
    
    oos_df = pd.DataFrame(oos_returns, index=dates, columns=shrinkage_list)

    sharpe_dict = {s: sharpe_ratio(oos_df[s]) for s in shrinkage_list}
    for s, sr in sharpe_dict.items():
        print(f"z = {s} -> Sharpe = {sr:.2f}")
    return oos_df, sharpe_dict

def produce_random_feature_managed_returns(P, stock_data, signals, num_seeds=10, scale = 1.0):
  """Generate *random Fourier features* and compute managed returns.

    For each seed:
      • Draw ``ω ~ 𝒩(0, 2/d)``
      • Create ``√2·sin(Sωᵀ)`` and ``√2·cos(Sωᵀ)`` feature blocks.
      • Collapse the cross‑section to a time‑series using
        :func:`build_managed_returns`.

    Parameters
    ----------
    P : int
        Number of random features per *sin/cos* block (total 2P post‑concat).
    stock_data : DataFrame with column ``'r_1'`` and MultiIndex *(id, date)*.
    signals : DataFrame with the same MultiIndex as *stock_data*.
    num_seeds : int, default 10
        Re‑draw *ω* this many times to enlarge the feature set.
    scale : float, default 1.0
        σ scaling on the random weight distribution.

    Returns
    -------
    DataFrame
        Managed‑return matrix whose columns are the random features generated
        across all seeds.
    """
  all_random_feature_managed_returns = pd.DataFrame()
  d = signals.shape[1]

  for seed in range(num_seeds):
    # every seed gives me a new chunk of factors
    np.random.seed(seed)
    omega = scale * np.sqrt(2) * np.random.randn(P, d) / np.sqrt(d)
    ins_sin = np.sqrt(2) * np.sin(signals @ omega.T) # signals @ \Theta are (NT) \times P dimensional.
    ins_cos = np.sqrt(2) * np.cos(signals @ omega.T) # signals @ \Theta are (NT) \times P dimensional.
    random_features = pd.concat([ins_sin, ins_cos], axis=1)

    # Now, I collapse the N dimension.
    random_feature_managed_returns = build_managed_returns(returns=stock_data['r_1'], signals=random_features)
    # random_feature_managed_returns are now T \times P
    all_random_feature_managed_returns = pd.concat([all_random_feature_managed_returns, random_feature_managed_returns], axis=1)
  return all_random_feature_managed_returns

def regression_with_alpha_and_tstat(predicted_variable, explanatory_variables):
    """
    Parameters
    ----------
    predicted_variable : pd.Series
        Dependent variable *y*.
    explanatory_variables : pd.DataFrame
        Explanatory variables *X* (no constant; one will be added).

    Returns
    -------
    alpha : float
    alpha_tstat : float
    """
    x_ = sm.add_constant(explanatory_variables)
    y_ = predicted_variable
    z_ = x_.copy().astype(float)
    result = sm.OLS(y_.values, z_.values).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    
    alpha = result.params[0]         # intercept (alpha)
    alpha_tstat = result.tvalues[0]  # t-statistic of intercept

    return alpha, alpha_tstat

def hw_efficient_portfolio_oos_2(raw_factor_returns: pd.DataFrame, model: str):
    """Convenience wrap of :func:`hw_efficient_portfolio_oos` for one λ only.

    Uses shrinkage_list = [100] for quick experiments and prints model‑specific
    Sharpe ratios.
    """
    oos_returns = []
    dates = []
    shrinkage_list = [100] 

    for t in range(360,len(raw_factor_returns)):
        X_train = raw_factor_returns.iloc[t-360:t,:].values
        y_train = np.ones((X_train.shape[0], 1)) 
        X_test = raw_factor_returns.iloc[t,:].values
        beta, optimal = ridge_regr(signals=X_train,
                       labels=y_train,
                       future_signals=X_test,
                       shrinkage_list=shrinkage_list)
        oos_returns.append(optimal)
        dates.append(raw_factor_returns.index[t])
        
    
    oos_df = pd.DataFrame(oos_returns, index=dates, columns=shrinkage_list)

    for s in shrinkage_list:
        print(f"{model}: Sharpe {sharpe_ratio(oos_df[s]):.2f}")
        

    return oos_df