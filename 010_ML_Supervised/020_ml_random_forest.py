# Random Forest Classifier for Sales Opportunity 
# Quality Prediction
# Using sklearn's RandomForestClassifier
# This model predicts whether a sales opportunity is "Good" or "Not Good"
# based on various features in the dataset.
# The dataset is assumed to be in an Excel file named 'SalesOpportunityDataSet.xlsx'

# Note to self: Random Forests 
# Sees to be better than Decision Trees
# because they reduce overfitting by averaging multiple trees.
# I tried this and the usage of features seems more balanced.
# I have more confidence in this model than the decision tree.

# This is also supported by data where all feaature
# are used
# # See Data here 
# 3  Engagement_Score    0.275198
# 7   Time_to_Respond    0.232535
# 6       Prior_Deals    0.185275

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = '..\SalesOpportunityDataSet.xlsx'
data = pd.read_excel(file_path, sheet_name='SalesOpportunityDataSet')

df = pd.DataFrame(data)

# Drop Opportunity_ID (not useful for prediction)
df = df.drop('Opportunity_ID', axis=1)

# Encode categorical variables
label_encoders = {}
for col in ['Industry', 'Company_Size', 'Contact_Title', 'Product_Interest', 'Region', 'Prior_Deals']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop('Is_Good_Opportunity', axis=1) # Since this is what we are predicting
y = df['Is_Good_Opportunity'] # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train decision tree

# Parameters to note:
# n_estimators: Number of trees in the forest (default 100 is usually good)
# max_depth: Limits the depth of each tree to prevent overfitting (optional)
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance insight
importances = clf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(importance_df)

# Optional: Plot feature importances
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Example prediction

# Prepare new opportunity using the correct encoder for each feature
new_opportunity = [[
    label_encoders['Industry'].transform(['Healthcare'])[0],
    label_encoders['Company_Size'].transform(['Large'])[0],
    label_encoders['Contact_Title'].transform(['VP of Operations'])[0],
    85,
    label_encoders['Product_Interest'].transform(['AI Analytics'])[0],
    label_encoders['Region'].transform(['Northeast'])[0],
    label_encoders['Prior_Deals'].transform(['Yes'])[0],
    2.0
]]

# Convert to DataFrame with correct feature names
new_opportunity_df = pd.DataFrame(new_opportunity, columns=X.columns)
prediction = clf.predict(new_opportunity_df)
print(f"Predicted Opportunity: {'Good' if prediction[0] == 1 else 'Not Good'}")