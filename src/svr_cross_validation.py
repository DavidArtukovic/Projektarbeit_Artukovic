import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def svr_cross_validation(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    initial_days: int = 1095,
    period_days: int = 90,
    horizon_days: int = 730,
) -> pd.DataFrame:
    """
    Perform time-series cross-validation for an SVR model in the same style as Prophet's cross_validation,
    with both X- and y-scaling so that yhat predictions are inverse-transformed back to the original scale
    before being stored.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame indexed by datetime (dtype = datetime64[ns]), sorted in ascending order.
        Must contain `target_col` and all `feature_cols`.
    feature_cols : list of str
        List of column names to use as exogenous features (X variables).
    target_col : str
        Name of the target column (y).
    initial_days : int, default=1095
        Number of days to include in the initial training window.
    period_days : int, default=90
        Number of days to move the cutoff forward after each split.
    horizon_days : int, default=730
        Number of days to forecast after each cutoff.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'ds'         : Timestamp of each forecasted point
        - 'y'          : Actual value of the target at 'ds'
        - 'yhat'       : SVR-predicted value (inverse-scaled back to original y-scale)
        - 'cutoff'     : Timestamp of the cutoff (end of training window)

        Each row corresponds to one forecasted timestamp between (cutoff + 1 day)
        and (cutoff + horizon_days). Exactly analogous to Prophet's cross_validation output,
        except there are no confidence intervals.
    """
    # Ensure the DataFrame is sorted by its datetime index
    df = df.sort_index()
    idx = df.index

    # Convert days to Timedelta for convenience
    initial_td = pd.Timedelta(days=initial_days)
    period_td = pd.Timedelta(days=period_days)
    horizon_td = pd.Timedelta(days=horizon_days)

    # Collect all cutoffs (Prophet uses an expanding window: training starts at t_start and grows)
    t_start = idx.min()
    t_end = idx.max()
    cutoffs = []
    current_cutoff = t_start + initial_td

    while current_cutoff + horizon_td <= t_end:
        cutoffs.append(current_cutoff)
        current_cutoff = current_cutoff + period_td

    # Prepare a list to gather all forecast rows
    rows = []

    for cutoff in cutoffs:
        # ===== 1. SPLIT TRAIN / TEST =====
        # Training window: all data <= cutoff (expanding window)
        train_mask = idx <= cutoff
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values.reshape(-1, 1)

        # Test window: from (cutoff + 1 day) up to (cutoff + horizon_days)
        test_start = cutoff + pd.Timedelta(days=1)
        test_end = cutoff + horizon_td
        test_mask = (idx >= test_start) & (idx <= test_end)
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_col].values.reshape(
            -1, 1
        )  # actual y in original scale
        ds_test = df.loc[test_mask].index

        # If we don't have the full horizon of data, skip this cutoff
        if len(y_test) < horizon_days:
            # Prophet’s cross_validation would drop incomplete horizons automatically
            continue

        # ===== 2. FEATURE SCALING =====
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # ===== 3. TARGET SCALING =====
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(
            y_train
        ).ravel()  # flatten to 1D array for SVR
        # Note: We do NOT fit scaler_y on y_test, since we want to invert predictions afterwards

        # ===== 4. FIT SVR MODEL =====
        svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)  # You can tune hyperparameters here
        svr.fit(X_train_scaled, y_train_scaled)

        # ===== 5. FORECAST (scaled) =====
        yhat_test_scaled = svr.predict(X_test_scaled).reshape(-1, 1)

        # ===== 6. INVERSE-SCALE yhat =====
        yhat_test = scaler_y.inverse_transform(
            yhat_test_scaled
        ).ravel()  # back to original scale

        # ===== 7. COLLECT RESULTS =====
        # For each point in the test set (horizon), store (ds, y_actual, yhat, cutoff)
        # y_test is already in original scale, yhat_test is now inverse-scaled
        for dt, y_true_orig, y_pred_orig in zip(ds_test, y_test.ravel(), yhat_test):
            rows.append(
                {"ds": dt, "y": y_true_orig, "yhat": y_pred_orig, "cutoff": cutoff}
            )

    # Build the resulting DataFrame
    df_cv_svr = pd.DataFrame(rows)
    # Ensure columns are in the same order as Prophet's cross_validation (minus CI columns)
    df_cv_svr = df_cv_svr[["ds", "y", "yhat", "cutoff"]]
    return df_cv_svr


if __name__ == "__main__":
    # ===================================
    # Example Usage
    # ===================================
    # Assume you have a DataFrame 'df' with a DatetimeIndex (freq="D" or "H"), sorted ascending,
    # containing:
    #   - A target column "nodal_demand"
    #   - Some feature columns, e.g. ["temperature_Schnar", "dry_soil_10", "hour", "is_weekend", ...]
    #
    # Example:
    # df = pd.read_parquet("wasserverbrauch_stuttgart.parquet", engine="pyarrow")

    # Specify which columns are exogenous features:
    feature_cols = [
        "temperature_Schnar",
        "dry_soil_10",
        "hour",
        "is_weekend",
        "sunshine_duration_Schnar",
        "precipitation_Schnar",
    ]
    target_col = "nodal_demand"

    # ------------------------------------------------
    # 1) Expanding-Window CV (Prophet-Style: training grows)
    # ------------------------------------------------
    df_cv_expanding = svr_cross_validation(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        initial_days=1095,
        period_days=90,
        horizon_days=730,
    )
    print("Expanding-Window CV (Prophet-Style):")
    print(df_cv_expanding.head())

    # After obtaining df_cv_expanding or df_cv_fixed, you can compute MAPE per horizon-day, plot results, etc.
