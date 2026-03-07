from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Allow frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
with open('fraud_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the input data structure (Expand this to match all your model's required features)
class ClaimData(BaseModel):
    age: int
    policy_annual_premium: float
    total_claim_amount: float
    # Add the rest of your features here (e.g., injury_claim, property_claim, etc.)

@app.post("/predict")
async def predict_fraud(data: ClaimData):
    # Convert input data to a DataFrame, as expected by the scikit-learn model
    input_df = pd.DataFrame([data.dict()])
    
    # Make the prediction (0 = Not Fraud, 1 = Fraud)
    prediction = model.predict(input_df)
    
    # Return the result
    result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
    return {"prediction": result}

# Run the server using: uvicorn main:app --reload