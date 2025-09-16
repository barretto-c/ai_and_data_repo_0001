# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
