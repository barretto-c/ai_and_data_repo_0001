# Flight Data Anomaly Detection using Isolation Forest
# This script detects anomalies in flight data using the Isolation Forest algorithm.

# After runign script you can use MLflow UI to visualize the results and metrics:
# mlflow ui

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns


# Load the dataset
file_path = '..\AircraftADSBData.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

with mlflow.start_run(run_name="Flight Anomaly Detection"):
	# Log parameters
	mlflow.log_param("contamination", 0.1)
	mlflow.log_param("random_state", 42)

	# Select features for anomaly detection
	features = ['Latitude', 'Longitude', 'Altitude', 'Ground Speed', 'Heading']
	X = data[features].copy()

	# Standardize numerical features
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	# Apply Isolation Forest for anomaly detection
	iso_forest = IsolationForest(contamination=0.1, random_state=42)  # Expect ~10% of data as anomalies
	data['Anomaly'] = iso_forest.fit_predict(X_scaled)
	data['Anomaly_Label'] = data['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

	# Analyze anomalies
	print("\nAnomaly Detection Results:")
	print(data[data['Anomaly'] == -1][['ICAO', 'Callsign', 'Timestamp', 'Latitude', 'Longitude', 'Altitude', 'Ground Speed', 'Heading']])

	# Summary of anomalies vs. normal points
	print("\nSummary of Anomalies vs. Normal Points:")
	print(data.groupby('Anomaly_Label')[['Latitude', 'Longitude', 'Altitude', 'Ground Speed', 'Heading']].mean())

	# Visualize anomalies (Altitude vs. Ground Speed)
	plt.figure(figsize=(8, 5))
	sns.scatterplot(x=X['Altitude'], y=X['Ground Speed'], hue=data['Anomaly_Label'], palette={'Normal': 'green', 'Anomaly': 'red'}, s=100)
	plt.xlabel('Altitude (feet)')
	plt.ylabel('Ground Speed (knots)')
	plt.title('Anomaly Detection: Altitude vs. Ground Speed')
	plt.legend(title='Status')
	plt.grid(True)
	plt.show()

	# Log metrics and artifacts
	num_anomalies = (data['Anomaly'] == -1).sum()
	mlflow.log_metric("num_anomalies", num_anomalies)
	output_path = 'output/flights_with_anomalies.csv'
	data.to_csv(output_path, index=False)
	mlflow.log_artifact(output_path)
	mlflow.sklearn.log_model(iso_forest, "isolation_forest_model")
	print("\nDataset with anomaly labels saved to 'flights_with_anomalies.csv'")