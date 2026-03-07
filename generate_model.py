import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

print("Loading data...")
# Load the dataset
df = pd.read_csv("insurance_claims.csv")

# Select the 3 features we are using in our HTML UI
features = ['age', 'policy_annual_premium', 'total_claim_amount']
X = df[features]

# Fill any missing values just in case
X = X.fillna(X.median())

# Encode the target variable ('Y'/'N' to 1/0)
le = LabelEncoder()
y = le.fit_transform(df['fraud_reported'])

print("Training model for deployment...")
# Train the model
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Save the model
filename = 'fraud_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rf, file)

print("✅ SUCCESS: fraud_model.pkl has been created successfully!")