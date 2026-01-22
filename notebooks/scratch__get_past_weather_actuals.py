# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: nyc_taxi (3.12.9)
#     language: python
#     name: python3
# ---

# %%
from datetime import date
import matplotlib.pyplot as plt
import meteostat as ms

# Specify location and time range
POINT = ms.Point(40.7812, -73.9665)  # central park lat / log
START = date(2018, 1, 1)
END = date(2025, 12, 31)

# Get nearby weather stations
stations = ms.stations.nearby(POINT, limit=20)
stations

# %%
# Get nearby weather stations
stations_out = ms.stations.nearby(POINT, limit=4)

# Get daily data & perform interpolation
ts = ms.daily(stations_out, START, END)
df = ms.interpolate(ts, POINT).fetch()

# %%
# Set time period
start = date(2018, 1, 1)
end = date(2018, 12, 31)

# Get daily data
ts = ms.daily(ms.Station(id="72502"), start, end)  # Newark Airport
df = ts.fetch()

# Print DataFrame
print(df)

# %%
# get station metadata
station = ms.stations.meta("72503")  # LaGuardia Airport

print(station)

# %%
# Get daily data
ts = ms.daily(ms.Station(id="72503"), start, end)  # Newark Airport
df = ts.fetch()

# Print DataFrame
print(df)

# %%
# Get daily data
ts = ms.daily(ms.Station(id="KTEB0"), start, end)  # Teterboro Airport
df = ts.fetch()

# Print DataFrame
print(df)

# %%
################################################################
# previous version of meteostat. API just updated to v2.0.1
################################################################
from datetime import datetime
import pandas as pd
from meteostat import Hourly, Daily, Stations


def get_station_by_icao(icao: str):
    """Get station metadata using ICAO code"""
    stations = Stations().icao(icao)
    st = stations.fetch(1)
    if len(st) == 0:
        raise ValueError(f"No station found for ICAO={icao}")
    return st.index[0]


def fetch_hourly(station_id, start, end):
    """Pull hourly Meteostat data"""
    data = Hourly(station_id, start, end)
    df = data.fetch()
    df = df.sort_index()
    return df


def fetch_daily(station_id, start, end):
    """Pull daily Meteostat data"""
    data = Daily(station_id, start, end)
    df = data.fetch()
    df = df.sort_index()
    return df


def find_hourly_gaps(df):
    df = df.copy()
    df["expected_next"] = df.index.shift(1, freq="H")
    df["actual_next"] = df.index
    df["gap"] = df["actual_next"] != df["expected_next"]
    gaps = df[df["gap"]].index
    return list(gaps)


def find_daily_gaps(df):
    df = df.copy()
    df["expected_next"] = df.index.shift(1, freq="D")
    df["actual_next"] = df.index
    df["gap"] = df["actual_next"] != df["expected_next"]
    gaps = df[df["gap"]].index
    return list(gaps)


START = datetime(2018, 1, 1)
end = datetime.now()

STATIONS = {
    "central_park": "KNYC",  # Central Park
    "la_guardia": "KLGA",  # LaGuardia Airport
}

###########################################################
# Workflow
###########################################################
results = {}

for name, icao in STATIONS.items():
    station_id = get_station_by_icao(icao)

    hourly_df = fetch_hourly(station_id, START, end)
    # hourly_gaps = find_hourly_gaps(hourly_df)
    # results[f"{name}_hourly"] = (hourly_df, hourly_gaps)
    results[f"{name}_hourly"] = hourly_df

    daily_df = fetch_daily(station_id, START, end)
    # daily_gaps = find_daily_gaps(daily_df)
    # results[f"{name}_daily"] = (daily_df, daily_gaps)
    results[f"{name}_hourly"] = hourly_df

results["central_park_hourly"][0].to_csv("knyc_hourly_2018_present.csv")
results["la_guardia_hourly"][0].to_csv("klga_hourly_2018_present.csv")
results["central_park_daily"][0].to_csv("knyc_daily_2018_present.csv")
results["la_guardia_daily"][0].to_csv("klga_daily_2018_present.csv")


# pd.Series(results["KNYC_hourly"][1]).to_csv("knyc_hourly_gaps.csv", index=False)
# pd.Series(results["KLGA_hourly"][1]).to_csv("klga_hourly_gaps.csv", index=False)
# pd.Series(results["KNYC_daily"][1]).to_csv("knyc_daily_gaps.csv", index=False)
# pd.Series(results["KLGA_daily"][1]).to_csv("klga_daily_gaps.csv", index=False)


# %%

# %%
from datetime import datetime
import pandas as pd
from meteostat import Hourly, Daily, Stations


# %%
def fetch_daily(station_id, start, end):
    """Pull daily Meteostat data"""
    data = Daily(station_id, start, end)
    df = data.fetch()
    df = df.sort_index()
    return df


# %%
START = datetime(2018, 1, 1)
end = datetime.now()

# %%
daily_df = fetch_daily("USW00094728", START, end)

# %%
daily_df

# %%

from datetime import date
import matplotlib.pyplot as plt
import meteostat as ms

# Specify location and time range
POINT = ms.Point(40.7812, -73.9665)
START = date(2018, 1, 1)
END = date(2018, 12, 31)

# Get nearby weather stations
stations_out = ms.stations.nearby(POINT, limit=4)

# Get daily data & perform interpolation
ts = ms.daily(stations_out, START, END)
df = ms.interpolate(ts, POINT).fetch()

# %%
# Set time period
start = date(2018, 1, 1)
end = date(2018, 12, 31)

# Get daily data
ts = ms.daily(ms.Station(id="10637"), start, end)
df = ts.fetch()

# Print DataFrame
print(df)

# %%
stations_out

# %%
df

# %%
