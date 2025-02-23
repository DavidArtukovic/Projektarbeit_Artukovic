import numpy as np
import pandas as pd
import h5py


def load_and_prepare_water_data(h5_path):
    """Loads water demand data from an HDF5 file and processes it.

    Args:
        h5_path (str): Path to the HDF5 file containing water demand data.

    Returns:
        pd.DataFrame: Processed water demand data with hourly resolution.
    """
    with h5py.File(h5_path, "r") as system_demand:
        group = system_demand["system_demand"]
        n_nodal_datapoints_available = group["n_nodal_datapoints_available"][:]
        nodal_demand = group["nodal_demand_sum"][:]
        system_time = group["time"][:]

    # Convert time data into DataFrame
    system_time_transposed = np.transpose(system_time)
    df = pd.DataFrame(
        system_time_transposed,
        columns=["year", "month", "day", "hour", "minute", "second", "milisecond"],
    )
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]])
    df["nodal_demand"] = nodal_demand
    df["n_nodal_datapoints_available"] = n_nodal_datapoints_available
    df.set_index("datetime", inplace=True)

    # Resample to hourly data
    df_hourly = df.resample("h").median()
    df_hourly.drop(
        columns=["year", "month", "day", "hour", "minute", "second", "milisecond"],
        inplace=True,
    )
    df_hourly["datetime"] = df_hourly.index

    return df_hourly


def load_and_prepare_weather_data():
    """Loads and processes weather data from two Stuttgart sources.

    Returns:
        pd.DataFrame: Merged and processed weather data with hourly resolution.
    """
    # Load temperature data from two locations
    temperature_schnarrenberg = pd.read_csv(
        "data/weather_climate/Temperaturdaten_Schnarrenberg_stuendlich.csv",
        index_col="MESS_DATUM",
        delimiter=";",
        parse_dates=["MESS_DATUM"],
        date_format="%Y%m%d%H",
    ).rename(columns={"TT_TU": "temperature_Schnar"})

    temperature_echterdingen = pd.read_csv(
        "data/weather_climate/Temperaturdaten_Echterdingen_stuendlich.csv",
        index_col="MESS_DATUM",
        delimiter=";",
        parse_dates=["MESS_DATUM"],
        date_format="%Y%m%d%H",
    ).rename(columns={"TT_TU": "temperature_Echt"})

    temperature = pd.merge(
        temperature_schnarrenberg,
        temperature_echterdingen,
        left_index=True,
        right_index=True,
        how="outer",
    )

    # Load sunshine duration data
    sun_schnarrenberg = pd.read_csv(
        "data/weather_climate/Sonnenschein_Schnarrenberg_stuendlich.csv",
        index_col="MESS_DATUM",
        delimiter=";",
        parse_dates=["MESS_DATUM"],
        date_format="%Y%m%d%H",
    ).rename(columns={"SD_SO": "sunshine_duration_Schnar"})

    sun_echterdingen = pd.read_csv(
        "data/weather_climate/Sonnenschein_Echterdingen_stuendlich.csv",
        index_col="MESS_DATUM",
        delimiter=";",
        parse_dates=["MESS_DATUM"],
        date_format="%Y%m%d%H",
    ).rename(columns={"SD_SO": "sunshine_duration_Echt"})

    sunshine = pd.merge(
        sun_schnarrenberg,
        sun_echterdingen,
        left_index=True,
        right_index=True,
        how="outer",
    )

    # Load precipitation data from two locations
    precipitation_schnarrenberg = pd.read_csv(
        "data/weather_climate/Niederschlag_Schnarrenberg_stuendlich.csv",
        index_col="MESS_DATUM",
        delimiter=";",
        parse_dates=["MESS_DATUM"],
        date_format="%Y%m%d%H",
    ).rename(columns={"  R1": "precipitation_Schnar"})

    precipitation_echterdingen = pd.read_csv(
        "data/weather_climate/Niederschlag_Echterdingen_stuendlich.csv",
        index_col="MESS_DATUM",
        delimiter=";",
        parse_dates=["MESS_DATUM"],
        date_format="%Y%m%d%H",
    ).rename(columns={"  R1": "precipitation_Echt"})

    precipitation = pd.merge(
        precipitation_schnarrenberg,
        precipitation_echterdingen,
        left_index=True,
        right_index=True,
        how="outer",
    )

    # Load soil moisture data
    soil = pd.read_csv(
        "data/weather_climate/derived_germany_soil_daily_historical_4928.csv",
        delimiter=";",
        parse_dates=["Datum"],
        date_format="%Y-%m-%d",
    ).rename(columns={"Datum": "date"})
    soil = soil[["date", "BF10", "BF20", "BF30", "BF40", "BF50", "BF60"]]
    soil["date"] = pd.to_datetime(soil["date"])

    # Merge all weather data into a single DataFrame
    weather = pd.merge(
        temperature, sunshine, left_index=True, right_index=True, how="outer"
    )
    weather = pd.merge(
        weather, precipitation, left_index=True, right_index=True, how="outer"
    )
    weather.index = pd.to_datetime(weather.index, format="%Y%m%d%H")
    weather["datetime"] = weather.index  # keep datetime as info
    weather["date"] = pd.to_datetime(weather.index.date)

    # Ensure both date columns are datetime type before merging
    weather = pd.merge(weather, soil, on="date", how="left")
    # weather = weather.set_index("datetime")
    weather.replace(-999, np.nan, inplace=True)  # Replace missing values with NaN

    columns_needed = [
        "datetime",
        "temperature_Schnar",
        "temperature_Echt",
        "sunshine_duration_Schnar",
        "sunshine_duration_Echt",
        "precipitation_Schnar",
        "precipitation_Echt",
        "BF10",
        "BF20",
        "BF30",
        "BF40",
        "BF50",
        "BF60",
    ]

    # Select only available columns to avoid KeyError
    weather = weather[[col for col in columns_needed if col in weather.columns]]
    weather = weather.reset_index(drop=True)

    return weather


if __name__ == "__main__":
    # Process and save water demand data
    df_water = load_and_prepare_water_data("data/water_demand/system_demand.h5")
    df_water.to_csv("data/water_demand/prepared_water_demand_data.csv", index=False)

    # Process and save weather data
    df_weather = load_and_prepare_weather_data()
    df_weather.to_csv("data/weather_climate/prepared_weather_data.csv", index=False)

    print("Data successfully prepared and saved.")
