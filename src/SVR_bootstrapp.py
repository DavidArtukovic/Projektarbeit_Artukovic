import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

def svr_bootstrap_ci(X_train, y_train, X_test, n_bootstraps=1000, alpha=0.05, **svr_params):
    """
    Computes bootstrapped confidence intervals for SVR predictions.
    """
    predictions = np.zeros((n_bootstraps, len(X_test)))

    for i in range(n_bootstraps):
        X_res, y_res = resample(X_train, y_train)
        model = SVR(**svr_params)
        model.fit(X_res, y_res)
        predictions[i] = model.predict(X_test)      
        if i == 0 or (i + 1) % 100 == 0:
            print(f"Iteration {i + 1} of {n_bootstraps} completed.")

    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - (alpha / 2))

    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)
    median_pred = np.median(predictions, axis=0)

    return median_pred, lower_bound, upper_bound

def run_svr_and_save(X_train_scaled, y_train_scaled, X_test_scaled, inverse_scaler, output_csv_path):
    """
    Runs the SVR bootstrap CI estimation and saves the result to a CSV file.
    """
    median_pred, lower_bound, upper_bound = svr_bootstrap_ci(
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        n_bootstraps=500,
        alpha=0.05,
        kernel="rbf",
        C=10,
        gamma=0.1,
        epsilon=0.1,
    )
    median_inv = inverse_scaler.inverse_transform(median_pred.reshape(-1, 1)).flatten()
    lower_inv = inverse_scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
    upper_inv = inverse_scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()

    # Build a DataFrame and save
    df_result = pd.DataFrame({
        "median_prediction": median_inv,
        "lower_bound": lower_inv,
        "upper_bound": upper_inv
    })
    df_result.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")

if __name__ == "__main__":
    import os

    # Prepare DataFrame
    SVR_df = pd.read_csv("Projektarbeit_Artukovic-1/data/SVR_df.csv", parse_dates=["datetime"])

    SVR_df.set_index("datetime", inplace=True)

    # split train and test
    train = SVR_df.loc[:"2021-12-31"]
    test = SVR_df.loc["2022-01-01":]

    X_train = train.drop("nodal_demand", axis=1)
    y_train = train["nodal_demand"]

    X_test = test.drop("nodal_demand", axis=1)
    y_test = test["nodal_demand"]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()

    # Run and save predictions
    run_svr_and_save(X_train_scaled, y_train_scaled, X_test_scaled, scaler_y, "Projektarbeit_Artukovic-1/data/svr_predictions_v2.csv")
