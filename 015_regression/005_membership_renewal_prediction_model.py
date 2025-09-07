# Membership Renewal Prediction Model
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Load and preprocess membership data
df = pd.read_csv("membership_data.csv")

# Features relevant to membership renewal prediction
features = [
    'membership_type',           # Basic, Premium, VIP
    'years_as_member',          # How long they've been a member
    'events_attended',          # Number of events attended this year
    'benefits_used',            # Number of member benefits utilized
    'payment_history',          # On-time payment record (0-1 scale)
    'engagement_score',         # Interaction with association activities
    'age_group',               # Member age category
    'location_type',           # Urban, Suburban, Rural
    'referrals_made'           # Number of new members they referred
]

target = 'will_renew'  # Binary: 1 = will renew, 0 = will not renew

# Pre-Procssing Encode categorical features
categorical_features = ['membership_type', 'age_group', 'location_type']
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store for future use

# Split data before preprocessing to prevent data leakage
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# A tensor is PyTorch's fundamental data structure
# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
print("Training data shape:", X_train_tensor)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define membership renewal prediction model
class MembershipRenewalPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16], dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer with sigmoid for binary classification
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Initialize model
model = MembershipRenewalPredictor(input_size=len(features))
criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with validation tracking
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
patience = 20

print("Training Membership Renewal Prediction Model...")
print("-" * 50)

for epoch in range(300):
    # Training phase
    model.train()
    optimizer.zero_grad()
    train_output = model(X_train_tensor)
    train_loss = criterion(train_output, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_output = model(X_test_tensor)
        val_loss = criterion(val_output, y_test_tensor)
    
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_membership_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Load best model
model.load_state_dict(torch.load('best_membership_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    # Probability predictions
    prob_predictions = model(X_test_tensor).squeeze().numpy()
    
    # Binary predictions (threshold = 0.5)
    binary_predictions = (prob_predictions > 0.5).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, binary_predictions)
precision = precision_score(y_test, binary_predictions)
recall = recall_score(y_test, binary_predictions)
f1 = f1_score(y_test, binary_predictions)
auc = roc_auc_score(y_test, prob_predictions)

print("\n" + "="*50)
print("MEMBERSHIP RENEWAL PREDICTION RESULTS")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# Predict renewal probabilities for all members
with torch.no_grad():
    all_features_scaled = scaler.transform(df[features])
    all_tensor = torch.tensor(all_features_scaled, dtype=torch.float32)
    all_predictions = model(all_tensor).squeeze().numpy()

df['renewal_probability'] = all_predictions
df['predicted_renewal'] = (all_predictions > 0.5).astype(int)

# Members at risk of not renewing (low renewal probability)
at_risk_members = df[df['renewal_probability'] < 0.3].sort_values('renewal_probability')
high_retention_members = df[df['renewal_probability'] > 0.8].sort_values('renewal_probability', ascending=False)

print("\n" + "="*50)
print("MEMBERSHIP INSIGHTS")
print("="*50)

print(f"\nMembers at HIGH RISK of not renewing: {len(at_risk_members)}")
print("Top 5 at-risk members:")
if len(at_risk_members) > 0:
    risk_cols = ['member_id', 'membership_type', 'years_as_member', 'events_attended', 'renewal_probability']
    print(at_risk_members[risk_cols].head().to_string(index=False))

print(f"\nMembers with HIGH RETENTION probability: {len(high_retention_members)}")
print("Top 5 likely to renew:")
if len(high_retention_members) > 0:
    print(high_retention_members[risk_cols].head().to_string(index=False))

# Feature importance analysis (using simple gradient-based approach)
def get_feature_importance():
    model.eval()
    # Use a sample of data for gradient analysis
    sample_data = X_train_tensor[:100].requires_grad_()
    output = model(sample_data)
    
    # Calculate gradients
    gradients = torch.autograd.grad(outputs=output.sum(), inputs=sample_data)[0]
    importance = torch.abs(gradients).mean(0).detach().numpy()
    
    return importance

importance_scores = get_feature_importance()
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importance_scores
}).sort_values('importance', ascending=False)

print(f"\n" + "="*50)
print("FEATURE IMPORTANCE FOR RENEWAL PREDICTION")
print("="*50)
print(feature_importance.to_string(index=False))

# Actionable recommendations
print(f"\n" + "="*50)
print("ACTIONABLE RECOMMENDATIONS")
print("="*50)

print("\n1. IMMEDIATE ATTENTION NEEDED:")
print(f"   • {len(at_risk_members)} members have <30% renewal probability")
print("   • Focus retention efforts on these members immediately")

print("\n2. RETENTION STRATEGIES:")
print("   • Increase event attendance for at-risk members")
print("   • Promote underutilized member benefits")
print("   • Improve payment experience and communication")
print("   • Enhance member engagement programs")

print("\n3. MONITORING:")
print("   • Track engagement_score and events_attended monthly")
print("   • Monitor payment_history for early warning signs")
print("   • Regular check-ins with long-term members")

# Save results for association management
results_summary = {
    'total_members': len(df),
    'predicted_renewals': sum(df['predicted_renewal']),
    'at_risk_count': len(at_risk_members),
    'high_retention_count': len(high_retention_members),
    'model_accuracy': accuracy,
    'model_auc': auc
}

print(f"\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
for key, value in results_summary.items():
    if isinstance(value, float):
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    else:
        print(f"{key.replace('_', ' ').title()}: {value}")

# Export at-risk members for targeted outreach
at_risk_members.to_csv('at_risk_members.csv', index=False)
print(f"\n📁 At-risk members list saved to 'at_risk_members.csv'")
print("📁 Use this file for targeted retention campaigns!")