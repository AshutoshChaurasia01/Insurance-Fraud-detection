from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('fraud_model.pkl', 'rb') as file:
    model = pickle.load(file)


class ClaimData(BaseModel):
    age: int
    policy_annual_premium: float
    total_claim_amount: float
    

@app.post("/predict")
async def predict_fraud(data: ClaimData):
    
    input_df = pd.DataFrame([data.dict()])
    
    prediction = model.predict(input_df)
    

    result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
    return {"prediction": result}
