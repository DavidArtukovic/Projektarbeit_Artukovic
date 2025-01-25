import pandas as pd
import os
import matplotlib.pyplot as plt


def create_raw_data():
    df = pd.read_csv("data/Wasserverbrauch.csv", delimiter=";", header=0, decimal=",")
    for filename in os.listdir("data"):
        if filename.endswith(".csv") and filename != "Wasserverbrauch.csv":
            file_path = os.path.join("data/", filename)
            current_df = pd.read_csv(
                file_path, delimiter=";", thousands=".", decimal=","
            )
            df = pd.merge(df, current_df, on="Jahr")

    # Summe der Einwohnerzahlen berechnen
    columns_to_sum = [
        col
        for col in df.columns
        if col != "Wasserbereitstellung_Summe" and col != "Jahr"
    ]
    df["Summe_Einwohner"] = df[columns_to_sum].sum(axis=1)

    return df


if __name__ == "__main__":
    raw_data = create_raw_data()
    print("Head of the merged DataFrame:")
    print(raw_data.head())

    # Create a quick chart
    plt.figure(figsize=(10, 6))
    plt.scatter(
        raw_data["Wasserbereitstellung_Summe"],
        raw_data["Summe_Einwohner"],
    )
    plt.xlabel("Wasserverbrauch Summe")
    plt.ylabel("Einwohner")
    plt.title("Summe Einwohner vs Wasserbereitstellung Summe")
    plt.legend()
    plt.show()
