import time
import requests
import numpy as np
import pandas as pd

LAT, LON = 22.9734, 78.6569
PARAMETERS = ["T2M", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"]


def download_nasa_power_year(year, latitude=LAT, longitude=LON):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }
    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    parameter_data = payload["properties"]["parameter"]
    df = pd.DataFrame(parameter_data)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df = df[df.index.notna()].replace(-999, np.nan)
    return df.apply(pd.to_numeric, errors="coerce")


def download_environmental_data(start_year, end_year):
    frames = []
    for year in range(start_year, end_year + 1):
        try:
            frames.append(download_nasa_power_year(year))
            print(f"Downloaded {year}")
            time.sleep(0.2)
        except Exception as error:
            print(f"Skipping {year}: {error}")

    if not frames:
        raise RuntimeError("No NASA POWER data could be downloaded")

    weather = pd.concat(frames).sort_index()
    annual = weather.resample("YE").agg({
        "T2M": "mean",
        "PRECTOTCORR": "sum",
        "RH2M": "mean",
        "ALLSKY_SFC_SW_DWN": "mean",
    }).reset_index()
    annual["year"] = annual["index"].dt.year
    return annual.drop(columns="index")


if __name__ == "__main__":
    data = download_environmental_data(2000, 2024)
    data.to_csv("nasa_power_environmental_data.csv", index=False)
    print(data.head())
