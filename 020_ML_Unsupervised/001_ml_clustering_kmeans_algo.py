import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = '..\SalesOpportunityDataSet.xlsx'
data = pd.read_excel(file_path, sheet_name='SalesOpportunityDataSet')

df = pd.DataFrame(data)

# Select features for clustering
features = ['Engagement_Score', 'Time_to_Respond', 'Company_Size']
X = data[features].copy()

# Encode Company_Size (ordinal encoding)
X['Company_Size'] = X['Company_Size'].map({'Small': 1, 'Medium': 2, 'Large': 3})

# Standardize numerical features for scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow Method
# The elbown method helps to find the optimal number of clusters 
wcss = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
print(f"WCSS values for k=1 to 5: {wcss}")

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Fit K-Means with chosen k (assuming k=3 from elbow method)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters: Mean of numerical features
print("\nCluster Analysis (Mean Values):")
# Ensure Company_Size is numeric for aggregation
data['Company_Size'] = data['Company_Size'].map({'Small': 1, 'Medium': 2, 'Large': 3})
print(data.groupby('Cluster')[['Engagement_Score', 'Time_to_Respond', 'Company_Size']].mean())

# Analyze clusters: Most common categorical features
print("\nCluster Analysis (Most Common Categorical Features):")
print(data.groupby('Cluster')[['Industry', 'Region', 'Product_Interest', 'Prior_Deals']].agg(lambda x: x.mode()[0]))

# Cross-tabulation with Is_Good_Opportunity
print("\nCluster vs. Is_Good_Opportunity:")
print(pd.crosstab(data['Cluster'], data['Is_Good_Opportunity'], normalize='index'))

# Visualize clusters (Engagement_Score vs. Time_to_Respond)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X['Engagement_Score'], y=X['Time_to_Respond'], hue=data['Cluster'], palette='viridis', s=100)
plt.xlabel('Engagement Score')
plt.ylabel('Time to Respond (hours)')
plt.title('Clusters of Opportunities')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Optional: Save the dataset with cluster labels
data.to_csv('output/opportunities_with_clusters.csv', index=False)
print("\nDataset with cluster labels saved to 'opportunities_with_clusters.csv'")