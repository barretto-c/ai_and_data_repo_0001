import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Matplotlib Demo using ADSB Signal Data")
# Load ADSB-like data
file_path = '..\AircraftADSBData.xlsx'
data = pd.read_excel(file_path)
print("Data Loaded from Excel:")
print("Columns in DataFrame:", data.columns.tolist())
# Display first few rows

print(data.head())

# Plot Altitude vs Timestamp for each ICAO
plt.figure(figsize=(10, 6))
for icao in data['ICAO'].unique():
    subset = data[data['ICAO'] == icao]
    plt.plot(subset['Timestamp'], subset['Altitude'], label=str(icao))
plt.title('Altitude vs Timestamp for each ICAO')
plt.xlabel('Timestamp')
plt.ylabel('Altitude (feet)')
plt.legend(title='ICAO')
plt.grid(True)
plt.show()

# Plot Speed vs Altitude for each ICAO
plt.figure(figsize=(10, 6))
for icao in data['ICAO'].unique():
    subset = data[data['ICAO'] == icao]
    plt.scatter(subset['Ground Speed'], subset['Altitude'], label=str(icao), alpha=0.6)
plt.title('Speed vs Altitude for each ICAO')
plt.xlabel('Speed (knots)')
plt.ylabel('Altitude (feet)')
plt.legend(title='ICAO')
plt.grid(True)
plt.show()
