# Prediction of sales opportunity quality using
# logistic regression

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (binary cross-entropy)
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(X @ weights)
    epsilon = 1e-5  # Avoid log(0)
    cost = -(1/m) * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
    return cost

# Gradient descent
def gradient_descent(X, y, weights, lr, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        h = sigmoid(X @ weights)
        gradient = (1/m) * (X.T @ (h - y))
        weights -= lr * gradient
        cost_history.append(compute_cost(X, y, weights))

    return weights, cost_history

# Prediction
def predict(X, weights, threshold=0.5):
    probs = sigmoid(X @ weights)
    return (probs >= threshold).astype(int)

# Example dataset (replace with your own)
df = pd.DataFrame({
    'Engagement_Score': [50, 80, 30, 90, 60],
    'Time_to_Respond': [5.0, 1.2, 7.5, 0.5, 3.0],
    'Prior_Deals': [1, 1, 0, 1, 0],
    'Is_Good_Opportunity': [1, 1, 0, 1, 0]
})

# Feature matrix and target
X = df[['Engagement_Score', 'Time_to_Respond', 'Prior_Deals']]
y = df['Is_Good_Opportunity']

# Normalize features
X = (X - X.mean()) / X.std()

# Add intercept term
X.insert(0, 'Intercept', 1)

# Convert to NumPy
X_np = X.values
y_np = y.values.reshape(-1, 1)
weights = np.zeros((X_np.shape[1], 1))

# Train model
weights, cost_history = gradient_descent(X_np, y_np, weights, lr=0.1, iterations=1000)

# Evaluate
y_pred = predict(X_np, weights)
accuracy = np.mean(y_pred == y_np)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_np, y_pred, target_names=["Not Good Opportunity", "Good Opportunity"]))
print(f"Weights: {weights.ravel()}")
# Predict a new sample
new_sample = pd.DataFrame([{
    'Engagement_Score': 70,
    'Time_to_Respond': 2.5,
    'Prior_Deals': 1
}])
new_sample = (new_sample - X.iloc[:, 1:].mean()) / X.iloc[:, 1:].std()  # Normalize
new_sample.insert(0, 'Intercept', 1)
new_sample_np = new_sample.values
predicted_class = predict(new_sample_np, weights)[0][0]

print(f"Predicted Class for New Sample: {'Good Opportunity' if predicted_class == 1 else 'Not Good Opportunity'}")

# Note: This is a simplified example. For real-world applications, consider using libraries like scikit-learn for robustness and efficiency.

