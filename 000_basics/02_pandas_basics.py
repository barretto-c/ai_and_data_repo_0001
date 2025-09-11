import pandas as pd

print("Pandas Basics Demo using ADSB Signal Data")

# Load ADSB-like data
file_path = '..\AircraftADSBData.xlsx'
data = pd.read_excel(file_path)
print("Data Loaded from Excel:")
print("Columns in DataFrame:", data.columns.tolist())

# Display first few rows
print(data.head())

# Display data types
print(data.dtypes)

# Summary statistics
print(data.describe())
# Check for missing values

print(data.isnull().sum())

print(data.head())

# Filter data for a specific aircraft type
filtered_data = data[data['ICAO'] == 'D4E5F6']
print("Filtered Data for ICAO 'Boeing D4E5F6':")
print(filtered_data)

# # Group by 'Type' and count occurrences
grouped_data = data.groupby('ICAO').size()
print("Grouped Data by Type:")
print(grouped_data)

# Sort data by 'Altitude'
sorted_data = data.sort_values(by='Altitude', ascending=False)
print("Data Sorted by Altitude:")
print(sorted_data.head())
