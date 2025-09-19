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

# Plot Information
plt.figure(figsize=(10, 6))
for icao in data['ICAO'].unique():
    subset = data[data['ICAO'] == icao]
    plt.plot(subset['Timestamp'], subset['Altitude'], label=str(icao))
plt.title('Time Lapse and Altitude for each ICAO')
plt.xlabel('Timestamp')
plt.ylabel('Altitude (feet)')
plt.legend(title='ICAO')
plt.grid(True)
plt.show()



