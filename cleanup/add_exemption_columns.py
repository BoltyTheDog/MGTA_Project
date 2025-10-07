import pandas as pd
from datetime import datetime

# Configuration variables
HStart: int = 11  # Regulation start hour
HEnd: int = 18    # Regulation end hour
DISTANCE_THRESHOLD_KM: float = 3500.0  # Distance threshold in KM for exemption
PUBLISHING_TIME: int = HStart  # Publishing time in hours (10am)

def parse_time_to_hours(time_str):
    try:
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        return time_obj.hour + time_obj.minute / 60.0
    except:
        return None

def is_exempt(row, distance_threshold, publishing_time,):
    # Condition 1: Check if Coming From is Non-ECAC
    if pd.notna(row['Coming From']) and str(row['Coming From']).strip().upper() == 'NON-ECAC':
        return "Yes"
    
    # Condition 2: Check if distance is larger than threshold
    if pd.notna(row['Distancia (km)']):
        try:
            distance = float(row['Distancia (km)'])
            if distance > distance_threshold:
                return "Yes"
        except (ValueError, TypeError):
            pass
        # Condition 3: Check ETD against publishing time
    if pd.notna(row['ETD']):
        etd_hours = parse_time_to_hours(str(row['ETD']))
        if etd_hours and etd_hours < publishing_time + 0.5:
            return "Yes"

    return "No"

def determine_delay_type(row, is_exempt, h_start, h_end):
        # Condition 3: Check if ETA (arrival time) is outside regulation hours
    if pd.notna(row['ETA']):
        eta_hours = parse_time_to_hours(str(row['ETA']))
        if eta_hours:
            # ETA is outside regulation window (before HStart or after HEnd)
            if eta_hours < h_start or eta_hours >= h_end:
                return "None"
            if is_exempt == "No":
                return "Ground"
            return "Air"

    
    return ""  # If ETD is missing or invalid

def add_exemption_columns(input_csv_path, output_csv_path, distance_threshold=DISTANCE_THRESHOLD_KM, 
                          h_start=HStart, h_end=HEnd, publishing_time=PUBLISHING_TIME):
    # Read the CSV with semicolon delimiter and handle encoding issues
    # Try common encodings: latin-1, iso-8859-1, cp1252
    try:
        df = pd.read_csv(input_csv_path, delimiter=';', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_csv_path, delimiter=';', encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(input_csv_path, delimiter=';', encoding='cp1252')
    
    # Add Exempt column
    df['Exempt'] = df.apply(lambda row: is_exempt(row, distance_threshold, publishing_time), axis=1)
    
    # Add Delay Type column based on exemption status and publishing time
    df['Delay Type'] = df.apply(lambda row: determine_delay_type(row, row['Exempt'], h_start, h_end), axis=1)
    
    # Save to new CSV
    df.to_csv(output_csv_path, sep=';', index=False)
    
    print(f"Processed {len(df)} flights")
    print(f"Exempt flights: {(df['Exempt'] == 'Yes').sum()}")
    print(f"Non-exempt flights: {(df['Exempt'] == 'No').sum()}")
    print(f"Output saved to: {output_csv_path}")
    
    return df

if __name__ == "__main__":
    # Input and output file paths (adjusted for cleanup folder location)
    input_file = "../Data/LEBL_10AUG2025_ECAC.csv"
    output_file = "../Data/LEBL_10AUG2025_ECAC_updated.csv"
    
    # Process the file
    df_updated = add_exemption_columns(
        input_file, 
        output_file,
        distance_threshold=DISTANCE_THRESHOLD_KM,
        h_start=HStart,
        h_end=HEnd,
        publishing_time=PUBLISHING_TIME
    )
    
    # Display sample of the updated data
    print("\nSample of updated data:")
    print(df_updated[['ARCID', 'Coming From', 'Distancia (km)', 'ETD', 'ETA', 'Exempt', 'Delay Type']].head(15))