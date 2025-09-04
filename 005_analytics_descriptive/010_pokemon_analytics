import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Timestamp for snapshot
timestamp = pd.Timestamp('2025-08-27 20:37:00')

# Example Pokémon names
pokemon_names = ['Pikachu', 'Charizard', 'Bulbasaur', 'Squirtle']

# Simulate Pokémon stats
num_rows = 4
data = {
    'Timestamp': [timestamp] * num_rows,
    'Name': pokemon_names,
    'HP': np.random.randint(35, 120, size=num_rows),
    'Attack': np.random.randint(40, 130, size=num_rows),
    'Defense': np.random.randint(35, 120, size=num_rows),
    'Speed': np.random.randint(30, 120, size=num_rows),
    'Type': np.random.choice(['Electric', 'Fire', 'Grass', 'Water'], size=num_rows)
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
    'HP': "HP",
    'Attack': "Attack",
    'Defense': "Defense",
    'Speed': "Speed"
}

# Display raw data
print("🔎 Simulated Pokémon Snapshot:\n")
print(df)

# Run diagnostics
run_diagnostics(df, columns_to_describe)

# Visualize HP and Attack
plt.figure(figsize=(8, 5))
plt.bar(df['Name'], df['HP'], color='limegreen', label='HP')
plt.plot(df['Name'], df['Attack'], color='crimson', marker='o', label='Attack')
plt.xlabel('Pokémon')
plt.ylabel('Stat Value')
plt.title('Pokémon HP and Attack')
plt.legend()
plt.tight_layout()
plt.show()
