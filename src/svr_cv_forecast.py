# -*- coding: utf-8 -*-
"""
SVR Expanding-Origin Cross-Validation & Forecast
================================================
This script performs:
 1. Expanding-origin rolling CV for SVR with manual feature & target scaling.
 2. Computes horizon-specific empirical 95% prediction intervals.
 3. Final two-year forecast (2022–2023) with manual scaling and PI.

Dependencies:
    pip install scikit-learn pandas numpy matplotlib

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------
# 1) Load data
# ---------------------------------------------------------------
df = pd.read_pickle("data/SVR_df.pkl")
df.index = pd.to_datetime(df.index)

y_col = "nodal_demand"
feature_cols = [c for c in df.columns if c != y_col]

# ---------------------------------------------------------------
# 2) Define rolling (expanding-origin) CV function
# ---------------------------------------------------------------
def rolling_cv_svr(
    df: pd.DataFrame,
    initial: pd.Timedelta,
    period: pd.Timedelta,
    horizon: pd.Timedelta,
    y_col: str,
    feature_cols: list[str],
    svr_kwargs: dict | None = None,
):
    svr_kwargs = svr_kwargs or {}
    df = df.sort_index()

    # build cutoffs
    cutoffs = []
    cutoff = df.index.min() + initial
    while cutoff + horizon <= df.index.max():
        cutoffs.append(cutoff)
        cutoff += period

    rows, metrics = [], []
    for fold, cutoff in enumerate(cutoffs, start=1):
        # split train/test
        train = df[df.index <= cutoff]
        test  = df[(df.index > cutoff) & (df.index <= cutoff + horizon)]
        print(f"Fold {fold}: train ≤ {cutoff.date()}, test ≤ {(cutoff+horizon).date()}")

        X_train = train[feature_cols]
        y_train = train[y_col]
        X_test  = test[feature_cols]
        y_test  = test[y_col]

        # manual scaling of X and y
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled  = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # fit SVR on scaled data
        model = SVR(**svr_kwargs)
        model.fit(X_train_scaled, y_train_scaled)

        # predict and inverse-transform
        yhat_scaled = model.predict(X_test_scaled)
        yhat = scaler_y.inverse_transform(yhat_scaled.reshape(-1, 1)).ravel()

        # collect point-wise predictions
        rows.append(pd.DataFrame({
            "ds":     test.index,
            "y":      y_test.values,
            "yhat":   yhat,
            "cutoff": cutoff
        }))

        # compute fold metrics
        metrics.append({
            "fold": fold,
            "cutoff": cutoff,
            "rmse": root_mean_squared_error(y_test, yhat),
            "mae":  mean_absolute_error(y_test, yhat),
            "mape": np.mean(np.abs((y_test.values - yhat) / y_test.values))
        })

    cv_df = pd.concat(rows, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)
    return cv_df, metrics_df

# ---------------------------------------------------------------
# 3) Run CV
# ---------------------------------------------------------------
initial = pd.Timedelta(days=1095)
period  = pd.Timedelta(days=30)
horizon = pd.Timedelta(days=730)
svr_params = {"kernel": "rbf", "C": 2, "epsilon": 0.1, "gamma": "scale"}

cv_df, fold_metrics = rolling_cv_svr(
    df, initial, period, horizon,
    y_col, feature_cols,
    svr_kwargs=svr_params
)

# ---------------------------------------------------------------
# 4) Build performance-metrics table
# ---------------------------------------------------------------
cv_df["horizon"] = (cv_df["ds"] - cv_df["cutoff"]).dt.days.astype(int)
cv_df.to_pickle("src/outputs/svr_cv_raw.pkl")

df_p = (
    cv_df.groupby("horizon").apply(
        lambda g: pd.Series({
            "mse":  np.mean((g.y - g.yhat)**2),
            "rmse": root_mean_squared_error(g.y, g.yhat),
            "mae":  mean_absolute_error(g.y, g.yhat),
            "mape": np.mean(np.abs((g.y - g.yhat) / g.y))
        })
    ).reset_index()
)

horizons = [90 * i for i in range(1, 9)]
df_p_sel = df_p[df_p["horizon"].isin(horizons)].reset_index(drop=True)
df_p_sel.to_latex(
    "src/outputs/performance_table_svr.tex",
    index=False,
    float_format="%.3f",
    caption="CV Performance of SVR at Different Horizons",
    label="tab:svr_performance"
)
print("→ performance_table_svr.tex written")

# ---------------------------------------------------------------
# 5) Final two-year forecast (2022–2023) with horizon-specific PI
# ---------------------------------------------------------------
# train/test split on original data
cutoff_final = pd.Timestamp("2021-12-31 23:00:00")
train = df[df.index <= cutoff_final]
test  = df[df.index >= "2022-01-01"]

X_train = train[feature_cols]
y_train = train[y_col]
X_future = df.loc[test.index, feature_cols]

# manual scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# fit on two-year training
svr = SVR(**svr_params)
svr.fit(X_train_scaled, y_train_scaled)

# predict and invert scale
X_future_scaled    = scaler_X.transform(X_future)
yhat_future_scaled = svr.predict(X_future_scaled)
yhat_future        = scaler_y.inverse_transform(yhat_future_scaled.reshape(-1,1)).ravel()

# compute residual quantiles by horizon from CV
resid_q_svr = (
    cv_df.groupby("horizon")["y"].apply(lambda g: None)  # placeholder
)
# reuse earlier residual/horizon code
cv_df["resid"]    = cv_df["y"] - cv_df["yhat"]
resid_q_svr = (
    cv_df.groupby("horizon")["resid"]
    .quantile([0.025, 0.975])
    .unstack(level=1)
    .rename(columns={0.025: "q_lo", 0.975: "q_hi"})
    .reset_index()
)

# assemble forecast DataFrame & attach horizon
forecast = pd.DataFrame({
    "ds":      test.index,
    "yhat":    yhat_future
})
forecast["horizon"] = (forecast["ds"] - cutoff_final).dt.days.astype(int)

# merge quantiles & build bounds
forecast = forecast.merge(resid_q_svr, on="horizon", how="left")
forecast["yhat_lower"] = forecast["yhat"] + forecast["q_lo"]
forecast["yhat_upper"] = forecast["yhat"] + forecast["q_hi"]
forecast = forecast[["ds","yhat","yhat_lower","yhat_upper"]]

# ---------------------------------------------------------------
# 6) Plot & save forecast
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,6))
hist = df[df.index >= "2017-01-01"]
ax.scatter(hist.index, hist[y_col], c="k", s=3, alpha=0.6, label="Actuals")
ax.plot(forecast["ds"], forecast["yhat"], label="SVR Forecast (Median)",alpha=0.6)
ax.fill_between(
    forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
    alpha=0.6, label="95% Interval"
)
ax.set_title("SVR Forecast and 95% Confidence Interval 2022–2023")
ax.set_xlabel("Date")
ax.set_ylabel("Nodal Demand")
ax.legend(loc="upper left")
ax.grid(linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("src/outputs/svr_forecast_95pi_2022-23.png", dpi=300)
plt.close(fig)
print("→ svr_forecast_95pi_2022-23.png written")
