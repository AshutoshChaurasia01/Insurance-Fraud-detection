import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

print("Loading data...")

df = pd.read_csv("insurance_claims.csv")

features = ['age', 'policy_annual_premium', 'total_claim_amount']
X = df[features]

X = X.fillna(X.median())

le = LabelEncoder()
y = le.fit_transform(df['fraud_reported'])

print("Training model for deployment...")

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

filename = 'fraud_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rf, file)

print("✅ SUCCESS: fraud_model.pkl has been created successfully!")
