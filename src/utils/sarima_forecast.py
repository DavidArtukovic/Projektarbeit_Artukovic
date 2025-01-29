import pmdarima as pmd
import numpy as np


def train_and_forecast_sarima(df, train_start, train_end, forecast_horizon=24):
    """
    Trains a SARIMA model and returns a 24-hour forecast.

    Parameters:
        df (pd.DataFrame): Time-indexed DataFrame with a 'nodal_demand' column.
        train_start (str or Timestamp): Start of training data.
        train_end (str or Timestamp): End of training data.
        forecast_horizon (int): Number of hours to forecast.

    Returns
    -------
    np.ndarray
        The 24-hour forecast as array.

    """

    df_train = df.loc[train_start:train_end]
    model = pmd.auto_arima(
        df_train["nodal_demand"],
        start_p=1,
        start_q=1,
        d=0,
        m=24,
        stationary=True,
        seasonal=True,
        trace=False,
        max_order=6,
        information_criterion="bic",
    )
    forecast = model.predict(n_periods=forecast_horizon)
    print(
        "order:", model.order, "x", model.seasonal_order, "; BIC:", model.bic().round(2)
    )
    return forecast
