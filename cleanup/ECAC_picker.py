import pandas as pd
import os
import csv

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
ecac_file = os.path.join(script_dir, "..", "Data", "ecac_complete_airports.csv")  # add .csv if needed
flight_file = os.path.join(script_dir, "..", "Data", "LEBL_10AUG2025.csv")
output_file = os.path.join(script_dir, "..", "Data", "LEBL_10AUG2025_updated.csv")

# --- Load ECAC airports CSV ---
ecac_df = pd.read_csv(ecac_file, header=None, names=['id', 'icao', 'continent', 'country', 'region'])
ecac_icao_set = set(ecac_df['icao'].str.strip())  # strip spaces

# --- Detect delimiter for flight CSV ---
with open(flight_file, 'r', newline='', encoding='utf-8') as f:
    sample = f.read(1024)
    dialect = csv.Sniffer().sniff(sample)
    delimiter = dialect.delimiter

# --- Load flight CSV ---
flight_df = pd.read_csv(flight_file, sep=delimiter)
flight_df.columns = flight_df.columns.str.strip()
print("Detected columns:", flight_df.columns.tolist())

# --- Add "Coming From" column ---
flight_df['Coming From'] = flight_df['ADEP'].apply(lambda x: 'ECAC' if str(x).strip() in ecac_icao_set else 'Non-ECAC')

# --- Save updated CSV ---
flight_df.to_csv(output_file, sep=delimiter, index=False)
print(f"Updated CSV saved as '{output_file}'")
