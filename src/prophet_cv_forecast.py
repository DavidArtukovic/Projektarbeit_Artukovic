# -*- coding: utf-8 -*-
"""
Prophet Expanding-Origin Cross-Validation & Forecast
===================================================

This script applies an expanding-origin rolling CV to a Prophet model, mirroring an SVR workflow.
External regressors and custom hyperparameters are taken from your `prophet_df.pkl`.

Steps
-----
1. Load `prophet_df.pkl` containing `ds`, `y`, and regressors.
2. Run rolling CV (`rolling_cv_prophet`) with initial/period/horizon.
3. Aggregate CV predictions into performance metrics and export to LaTeX.
4. Train final model on data ≤ 2021-12-31 and forecast 2022–2023 with an empirical 95% PI.
5. Save plots & outputs under `src/outputs/`.

Author: David Artukovic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Helper: RMSE
def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 1) Load pre-prepared DataFrame with ds, y, and regressors
# ---------------------------------------------------------
prophet_df = pd.read_pickle("data/prophet_df.pkl")  # ensure this file has columns: ds, y, and your regressors
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
prophet_df = prophet_df.set_index("ds").sort_index()

# Define target & regressors
feature_cols = [
    "temperature_Schnar",
    "sunshine_duration_Schnar",
    "BF10",
    "BF30",
    "T_hot",
    "T_hot^2",
    "is_holiday",
    "day_above_30",
]
y_col = "y"

# 2) Rolling CV for Prophet
def rolling_cv_prophet(
    df: pd.DataFrame,
    initial: pd.Timedelta,
    period: pd.Timedelta,
    horizon: pd.Timedelta,
    feature_cols: list[str],
    prophet_kwargs: dict | None = None,
):
    prophet_kwargs = prophet_kwargs or {}
    df = df.sort_index()

    # build cutoffs
    cutoffs = []
    cutoff = df.index.min() + initial
    while cutoff + horizon <= df.index.max():
        cutoffs.append(cutoff)
        cutoff += period

    rows, metrics = [], []
    for fold, cutoff in enumerate(cutoffs, start=1):
        print(f"Fold {fold}: train ≤ {cutoff.date()}, test ≤ {(cutoff+horizon).date()}")
        train = df[df.index <= cutoff]
        test = df[(df.index > cutoff) & (df.index <= cutoff + horizon)]

        # prepare frames
        df_train = train[feature_cols + [y_col]].reset_index().rename(columns={"ds":"ds","y":y_col})
        df_test  = test[feature_cols + [y_col]].reset_index().rename(columns={"ds":"ds","y":y_col})

        # fit Prophet
        m = Prophet(**prophet_kwargs)
        for reg in feature_cols:
            m.add_regressor(reg)
        m.fit(df_train)

        # forecast
        future = df_test[["ds"] + feature_cols]
        f = m.predict(future)

        y_true = df_test[y_col].values
        y_pred = f["yhat"].values

        # collect results
        rows.append(pd.DataFrame({
            "ds":    df_test["ds"],
            "y":     y_true,
            "yhat":  y_pred,
            "cutoff": cutoff
        }))
        metrics.append({
            "fold":   fold,
            "cutoff": cutoff,
            "rmse":   root_mean_squared_error(y_true, y_pred),
            "mae":    mean_absolute_error(y_true, y_pred),
            "mape":   np.mean(np.abs((y_true - y_pred) / y_true))
        })

    cv_df = pd.concat(rows, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)
    return cv_df, metrics_df

# Prophet hyperparameters
prophet_params = {
    "changepoint_prior_scale": 0.2,
    "weekly_seasonality": True,
    "daily_seasonality": True,
    "yearly_seasonality": False,
    "growth": "flat",
    "interval_width": 0.95,
}

# CV settings (3y init, 30d period, 2y horizon)
initial = pd.Timedelta(days=1095)
period  = pd.Timedelta(days=30)
horizon = pd.Timedelta(days=730)

cv_df, fold_metrics = rolling_cv_prophet(
    prophet_df,
    initial,
    period,
    horizon,
    feature_cols,
    prophet_kwargs=prophet_params
)

# 3) Performance table
cv_df["horizon"] = (cv_df["ds"] - cv_df["cutoff"]).dt.days.astype(int)
cv_df.to_pickle("src/outputs/prophet_cv_raw.pkl")

import matplotlib

# aggregate metrics
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

# build df_p

df_p = (
    cv_df.groupby("horizon").apply(
        lambda g: pd.Series({
            "mse":  np.mean((g.y - g.yhat)**2),
            "rmse": root_mean_squared_error(g.y, g.yhat),
            "mae":  mean_absolute_error(g.y, g.yhat),
            "mape": np.mean(np.abs((g.y - g.yhat)/g.y))
        })
    )
    .reset_index()
)
# select horizons
horizons = [90*i for i in range(1,9)]
df_p_sel = df_p[df_p.horizon.isin(horizons)].reset_index(drop=True)
# export

df_p_sel.to_latex(
    "src/outputs/performance_table_prophet.tex",
    index=False,
    float_format="%.3f",
    caption="CV Performance of Prophet at Different Horizons",
    label="tab:prophet_performance"
)
print("→ performance_table_prophet.tex written")

# 4) Final model & forecast (2022-2023)
cutoff_final = pd.Timestamp("2021-12-31 23:00:00")
train_final = prophet_df[prophet_df.index <= cutoff_final]

# prepare train frame
train_frame = train_final.reset_index()

m_final = Prophet(**prophet_params)
for reg in feature_cols:
    m_final.add_regressor(reg)
m_final.fit(train_frame)

# future frame
future_period = prophet_df[(prophet_df.index >= "2022-01-01") & (prophet_df.index <= "2023-12-31 23:00:00")]
future_frame = future_period[feature_cols].reset_index()
forecast = m_final.predict(future_frame)

# 1) compute residuals and horizon in the CV frame
cv_df["resid"] = cv_df.y - cv_df.yhat
cv_df["horizon"] = (cv_df["ds"] - cv_df["cutoff"]).dt.days.astype(int)

# 2) compute 2.5% and 97.5% quantiles *by horizon*
resid_q = (
    cv_df
    .groupby("horizon")["resid"]
    .quantile([0.025, 0.975])
    .unstack(level=1)
    .rename(columns={0.025: "q_lo", 0.975: "q_hi"})
    .reset_index()
)

# 3) assign horizon to your final forecast
forecast["horizon"] = (forecast["ds"] - cutoff_final).dt.days.astype(int)

# 4) merge quantiles onto forecast
out = (
    forecast[["ds", "yhat", "horizon"]]
    .merge(resid_q, on="horizon", how="left")
)

# 5) build lower/upper bounds
out["yhat_lower"] = out["yhat"] + out["q_lo"]
out["yhat_upper"] = out["yhat"] + out["q_hi"]

# 6) drop helper columns if you like
out = out[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# 5) Plot & save
fig, ax = plt.subplots(figsize=(12,6))
hist = prophet_df[prophet_df.index >= "2017-01-01"]
ax.scatter(hist.index, hist[y_col], c="k", s=3, alpha=0.5, label="Actuals")
ax.plot(out.ds, out.yhat, label="Prophet Forecast", color="#D55E00", alpha = 0.6)
ax.fill_between(out.ds, out.yhat_lower, out.yhat_upper, alpha=0.6, color="#F4CEA2", label="95% Interval")
ax.set_title("Prophet Forecast with 95% Interval (2022-2023)")
ax.set_xlabel("Date")
ax.set_ylabel("Nodal Demand")
ax.legend(loc="upper left")
ax.grid(linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("src/outputs/prophet_forecast_95pi_2022-23_v2.png", dpi=300)
# plt.close(fig)
print("→ prophet_forecast_95pi_2023.png written")
