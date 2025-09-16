import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to simulate scoring based on features
def score_lead(engagement, response_time, prior_deals):
    # Simple weighted sum (logit)
    return 0.05 * engagement - 0.3 * response_time + 1.0 * prior_deals

# Case 1: Likely Fraud
fraud_case = {
    'Industry': 'Unknown',
    'Company_Size': 'Small',
    'Contact_Title': 'Unverified',
    'Engagement_Score': 25,
    'Time_to_Respond': 0.3,
    'Prior_Deals': 0
}
fraud_logit = score_lead(fraud_case['Engagement_Score'], fraud_case['Time_to_Respond'], fraud_case['Prior_Deals'])
fraud_prob = sigmoid(fraud_logit)

print("Fraud Case")
print(f"Lead Details: {fraud_case}")
print(f"Raw Score: {fraud_logit:.2f}")
print(f"Predicted Probability of Legitimacy: {fraud_prob:.2f}")
print("Interpretation: Likely fraudulent\n")

# Case 2: Likely Legitimate
legit_case = {
    'Industry': 'Finance',
    'Company_Size': 'Medium',
    'Contact_Title': 'Risk Manager',
    'Engagement_Score': 85,
    'Time_to_Respond': 2.0,
    'Prior_Deals': 1
}
legit_logit = score_lead(legit_case['Engagement_Score'], legit_case['Time_to_Respond'], legit_case['Prior_Deals'])
legit_prob = sigmoid(legit_logit)

print("Legitimate Case")
print(f"Lead Details: {legit_case}")
print(f"Raw Score: {legit_logit:.2f}")
print(f"Predicted Probability of Legitimacy: {legit_prob:.2f}")
print("Interpretation: Likely legitimate")
