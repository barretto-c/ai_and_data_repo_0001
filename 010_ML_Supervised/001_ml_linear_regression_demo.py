# Prediction of time to respond to a sales opportunity
# Definition: Time to respond is the duration (in hours) between the initial contact with a sales opportunity and the first meaningful engagement (e.g., a follow-up call or meeting).
# Definition of Liner Regression: A statistical method that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. 
# It is used for predicting continuous outcomes.

#Update 9/18/2025
# Lasso (L1) gave the lowest Mean Squared Error and highest R², suggesting it’s capturing the signal more cleanly—possibly by zeroing out noisy or less relevant features.
# Ridge (L2) also improved over plain Linear Regression, but not as dramatically.
# L2
# Mean Squared Error: 12.73
# R² Score: 0.70

import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load dataset from Excel
file_path = '..\SalesOpportunityDataSet.xlsx'
data = pd.read_excel(file_path, sheet_name='SalesOpportunityDataSet')
print("Data Loaded from Excel:")
print(data.head())

# Step 3: Encode categorical features
# Encode categorical (text) columns into numeric values so they can be used in machine learning models
# For each listed column, create a LabelEncoder, fit it to the column, transform the text to numbers, and store the encoder for later use
label_encoders = {}
for col in ['Industry', 'Company_Size', 'Contact_Title', 'Product_Interest', 'Region', 'Prior_Deals']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 4: Split features and target
y = data['Time_to_Respond']
# Remove 'Opportunity_ID' from features as it is just an identifier

X = data.drop(['Time_to_Respond', 'Opportunity_ID'], axis=1)
y = data['Time_to_Respond']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 8: Predict a new sample
new_sample = pd.DataFrame([{
    'Industry': label_encoders['Industry'].transform(['Finance'])[0],
    'Company_Size': label_encoders['Company_Size'].transform(['Medium'])[0],
    'Contact_Title': label_encoders['Contact_Title'].transform(['Risk Manager'])[0],
    'Engagement_Score': 70,  # Use a realistic value from your data
    'Product_Interest': label_encoders['Product_Interest'].transform(['Risk Analytics'])[0],
    'Region': label_encoders['Region'].transform(['West'])[0],
    'Prior_Deals': label_encoders['Prior_Deals'].transform(['Yes'])[0],
    'Is_Good_Opportunity': 1  # Use a realistic value
}])
print(f"New Sample for Prediction: {new_sample}")

predicted_time = model.predict(new_sample)[0]
predicted_time = max(0, predicted_time)  # Ensure non-negative prediction
print(f"Predicted Time to Respond: {predicted_time:.2f} hours")


# 🔹 Ridge Regression (L2)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
print("\nRidge Regression Results:")
print(f"Mean Squared Error: {ridge_mse:.2f}")
print(f"R² Score: {ridge_r2:.2f}")

# 🔹 Lasso Regression (L1)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
print("\nLasso Regression Results:")
print(f"Mean Squared Error: {lasso_mse:.2f}")
print(f"R² Score: {lasso_r2:.2f}")

# 🔮 Predict using Ridge and Lasso
ridge_time = ridge_model.predict(new_sample)[0]
ridge_time = max(0, ridge_time)
print(f"Predicted Time to Respond (Ridge): {ridge_time:.2f} hours")

lasso_time = lasso_model.predict(new_sample)[0]
lasso_time = max(0, lasso_time)
print(f"Predicted Time to Respond (Lasso): {lasso_time:.2f} hours")