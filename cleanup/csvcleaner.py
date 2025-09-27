import pandas as pd

# load your CSV
df = pd.read_csv("Data/airports_world.csv", header=None, names=["id","icao","region","iso_country","subdivision"])

# ECAC ISO2 country codes
ecac_countries = ["AL","AM","AT","AZ","BE","BA","BG","HR","CY","CZ","DK","EE","FI","FR",
                  "GE","DE","GR","HU","IS","IE","IT","LV","LT","LU","MT","MC","MD","ME",
                  "NL","MK","NO","PL","PT","RO","RS","SK","SI","ES","SE","CH","TR","UA","GB"]

# filter
df_ecac = df[df["iso_country"].isin(ecac_countries)]

# save
df_ecac.to_csv("ecac_airports.csv", index=False)