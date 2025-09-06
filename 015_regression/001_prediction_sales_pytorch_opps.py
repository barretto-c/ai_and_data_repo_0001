import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and preprocess data
df = pd.read_csv("sales_data.csv")
features = ['category', 'region', 'price', 'previous_sales']
target = 'sales_opportunity'

# Encode categorical features
for col in ['category', 'region']:
    df[col] = LabelEncoder().fit_transform(df[col])

# Scale numerical features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Convert to tensors
X = torch.tensor(df[features].values, dtype=torch.float32)
y = torch.tensor(df[target].values, dtype=torch.float32).view(-1, 1)

# Define model
class SalesRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(features), 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SalesRanker()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Predict and rank
with torch.no_grad():
    predictions = model(X).squeeze()
    df['predicted_opportunity'] = predictions.numpy()
    ranked = df.sort_values(by='predicted_opportunity', ascending=False)

# Most salable product
most_salable = ranked.iloc[0]
print("Most salable product:", most_salable['product_id'])
print("Predicted opportunity score:", most_salable['predicted_opportunity'])
