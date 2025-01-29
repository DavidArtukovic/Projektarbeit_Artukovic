import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List


def create_yearly_df() -> pd.DataFrame:
    df = pd.read_csv(
        "../data/Wasserverbrauch.csv", delimiter=";", header=0, decimal=","
    )
    for filename in os.listdir("../data"):
        if filename.startswith("Einwohner"):
            file_path = os.path.join("../data/", filename)
            current_df = pd.read_csv(
                file_path, delimiter=";", thousands=".", decimal=","
            )
            df = pd.merge(df, current_df, on="Jahr")

    # Sum the inhabitants in Stuttgart area
    columns_to_sum: List[str] = [
        col
        for col in df.columns
        if col != "Wasserbereitstellung_Summe" and col != "Jahr"
    ]
    df["Summe_Einwohner"] = df[columns_to_sum].sum(axis=1)

    # Read Schnarrenberg yearly data
    schnarrenberg_df = pd.read_csv(
        "../data/Schnarrenberg_yearly.csv",
        delimiter=",",
        header=0,
        decimal=".",
        index_col=False,
    )

    # Extract year from Zeitstempel and create a new column 'Jahr'
    schnarrenberg_df["Jahr"] = pd.to_datetime(
        schnarrenberg_df["Zeitstempel"], format="%Y-%m-%d"
    ).dt.year

    # Merge Schnarrenberg yearly data with the main DataFrame on 'Jahr'
    df = pd.merge(df, schnarrenberg_df, on="Jahr", how="left")

    return df
