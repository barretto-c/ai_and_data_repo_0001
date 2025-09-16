import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = '..\SalesOpportunityDataSet.xlsx'
data = pd.read_excel(file_path, sheet_name='SalesOpportunityDataSet')

# Encode categorical features
label_encoders = {}
for col in ['Industry', 'Company_Size', 'Contact_Title', 'Product_Interest', 'Region', 'Prior_Deals']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


# Split features and target
X = data.drop(['Is_Good_Opportunity', 'Opportunity_ID'], axis=1)
y = data['Is_Good_Opportunity']

# Scale numerical features
# Standardizing features with StandardScaler ensures
# that each numerical input contributes equally to the 
# model, preventing bias toward variables with larger scales.
# This improves training stability and accuracy, especially for models
# sensitive to feature magnitude like logistic regression.
scaler = StandardScaler()
X[['Engagement_Score', 'Time_to_Respond']] = scaler.fit_transform(X[['Engagement_Score', 'Time_to_Respond']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(y_train)}, Testing samples: {len(y_test)}")
# Train Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Good Opportunity', 'Good Opportunity']))

# Predict a new sample
new_sample = pd.DataFrame([{
    'Industry': label_encoders['Industry'].transform(['Finance'])[0],
    'Company_Size': label_encoders['Company_Size'].transform(['Medium'])[0],
    'Contact_Title': label_encoders['Contact_Title'].transform(['Risk Manager'])[0],
    'Engagement_Score': 70,
    'Product_Interest': label_encoders['Product_Interest'].transform(['Risk Analytics'])[0],
    'Region': label_encoders['Region'].transform(['West'])[0],
    'Prior_Deals': label_encoders['Prior_Deals'].transform(['Yes'])[0],
    'Time_to_Respond': 2.5
}])

new_sample[['Engagement_Score', 'Time_to_Respond']] = scaler.transform(new_sample[['Engagement_Score', 'Time_to_Respond']])
predicted_prob = model.predict_proba(new_sample)[:, 1][0]
predicted_class = model.predict(new_sample)[0]
print(f"Predicted Probability (Good Opportunity): {predicted_prob:.2f}")
print(f"Predicted Class: {'Good Opportunity' if predicted_class == 1 else 'Not Good Opportunity'}")
