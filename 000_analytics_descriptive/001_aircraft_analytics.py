import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Timestamp for snapshot
timestamp = pd.Timestamp('2025-08-27 20:37:00')

# Aircraft ICAO codes
aircraft_ids = ['A1B2C3', 'D4E5F6']

# Altitude (ft)
## Unit: Feet above mean sea level (MSL)
# #Source: Often derived from barometric pressure sensors or GPS

#ICAO: Unique 24-bit address assigned to each aircraft
## Unit: Hexadecimal string
# #Source: Assigned by ICAO and used in ADS-B transmissions

# Ground Speed (kts)
## Unit: Knots (nautical miles per hour)    
# #Source: Calculated from GPS data or derived from airspeed and wind information

# Vertical Rate (fpm)
## Unit: Feet per minute
# #Source: Derived from changes in altitude over time, often using barometric or GPS data

# Position Confidence
## Unit: Dimensionless (0 to 1 scale)
# #Source: Calculated based on the quality of the position data, including factors like GPS signal strength and number of satellites


# Simulate telemetry
data = {
    'Timestamp': [timestamp] * 2,
    'ICAO': aircraft_ids,
    'Altitude_ft': np.random.normal(loc=35000, scale=200, size=2).round(),
    'GroundSpeed_kts': np.random.normal(loc=470, scale=15, size=2).round(1),
    'VerticalRate_fpm': np.random.choice([0, 64, -64, 128, -128], size=2),
    'PositionConfidence': np.random.uniform(0.92, 0.99, size=2).round(3)
}

# Create DataFrame
df = pd.DataFrame(data)

# Diagnostic function
def describe_column(series, label="Metric"):
    print(f"\nStatistics for: {label}")
    print("-" * 20)
    print(f"Count: {series.count()}")
    print(f"Mean: {series.mean():.2f}")
    print(f"Median: {series.median()}")
    print(f"Mode: {series.mode().values.tolist()}")
    print(f"Standard Deviation: {series.std():.2f}")
    print(f"Variance: {series.var():.2f}")
    print(f"Min: {series.min()}")
    print(f"Max: {series.max()}")
    print(f"Range: {series.max() - series.min():.2f}")
    print(f"25th Percentile (Q1): {series.quantile(0.25):.2f}")
    print(f"75th Percentile (Q3): {series.quantile(0.75):.2f}")
    print(f"IQR (Q3 - Q1): {series.quantile(0.75) - series.quantile(0.25):.2f}")
    print(f"Skewness: {series.skew():.2f}")
    print(f"Kurtosis: {series.kurtosis():.2f}")
    print("-" * 40)

# Batch diagnostics
def run_diagnostics(df, columns_with_labels):
    for col, label in columns_with_labels.items():
        if col in df.columns:
            describe_column(df[col], label=label)
        else:
            print(f"⚠️ Column '{col}' not found in DataFrame.")

# Define columns to analyze
columns_to_describe = {
    'Altitude_ft': "Altitude (ft)",
    'GroundSpeed_kts': "Ground Speed (kts)",
    'VerticalRate_fpm': "Vertical Rate (fpm)",
    'PositionConfidence': "Position Confidence"
}

# Display raw data
print("🛫 Simulated ADS-B Snapshot:\n")
print(df)

# Run diagnostics
run_diagnostics(df, columns_to_describe)

# Visualize Altitude and Ground Speed
plt.figure(figsize=(8, 5))
plt.bar(df['ICAO'], df['Altitude_ft'], color='skyblue', label='Altitude (ft)')
plt.plot(df['ICAO'], df['GroundSpeed_kts'], color='orange', marker='o', label='Ground Speed (kts)')
plt.xlabel('Aircraft ICAO')
plt.ylabel('Value')
plt.title('Aircraft Altitude and Ground Speed')
plt.legend()
plt.tight_layout()
plt.show()
