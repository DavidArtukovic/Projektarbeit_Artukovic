import numpy as np
import pandas as pd


def fourier_series(t, K, period):
    """
    t: DatetimeIndex or range of time steps
    K: number of sine/cosine pairs
    period: e.g., 168 (hours in a week)
    """
    x = np.arange(len(t))
    data = {}
    for k in range(1, K + 1):
        data[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * x / period)
        data[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * x / period)

    return pd.DataFrame(data, index=t)
