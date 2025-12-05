import pandas as pd
import numpy as np

def compute_ewma_vol(rets, lambda_=0.94):
    """
    Compute EWMA volatility for each asset.
    rets: DataFrame (date x asset) of daily returns.
    lambda_: decay factor (0.94 for daily, 0.97+ for slower memory).

    Returns:
        ewma_vol: DataFrame of annualized volatilities (same shape as rets).
    """
    # daily variance recursion: σ²_t = λ σ²_{t-1} + (1−λ) r²_t
    ewma_var = rets.copy() * 0

    # initialize with unconditional variance per asset
    init_var = rets.var()
    ewma_var.iloc[0] = init_var

    for t in range(1, len(rets)):
        ewma_var.iloc[t] = (
            lambda_ * ewma_var.iloc[t-1] +
            (1 - lambda_) * (rets.iloc[t] ** 2)
        )

    # annualize: multiply variance by 252 and take sqrt
    ewma_vol = np.sqrt(ewma_var * 252)

    return ewma_vol

def compute_garch_vol(rets, omega=1e-6, alpha=0.05, beta=0.90):
    """
    Compute GARCH(1,1) volatility for each asset.
    r_t^2 = conditional variance input
    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Returns:
        garch_vol: DataFrame of annualized volatilities (same shape as rets)
    """
    garch_var = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)

    # initialize variance with unconditional variance
    init_var = rets.var()
    garch_var.iloc[0] = init_var

    for t in range(1, len(rets)):
        r2_prev = rets.iloc[t-1] ** 2
        garch_var.iloc[t] = omega + alpha * r2_prev + beta * garch_var.iloc[t-1]

    # annualize
    garch_vol = (garch_var * 252).pow(0.5)
    return garch_vol
