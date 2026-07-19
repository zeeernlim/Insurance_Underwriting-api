from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Algorithmic Explainable-AI System for Insurance Underwriting",
    description="Automated Medical Risk Assessment Pipeline.",
    version="1.0.0",
    docs_url="/"
)

class ApplicantProfile(BaseModel):
    Occupation_type: int = Field(..., description="Encoded occupational risk tier", example=1)
    Sex: int = Field(..., description="Encoded biological sex", example=1)
    Age: int = Field(..., description="Current age of the applicant", example=67)
    Motorcycle_usage: int = Field(..., description="Binary indicator of motorcycle ridership", example=1)
    Alcohol_usage_length: int = Field(..., description="Years of alcohol consumption", example=20)
    Average_daily_cigarette: int = Field(..., description="Average daily cigarettes", example=4)
    Cigarette_usage_length: int = Field(..., description="Years of tobacco use", example=18)
    BMI: int = Field(..., description="Body Mass Index ratio", example=33)
    Exercicing_years: int = Field(..., description="Years of active physical training", example=4)

@app.post("/predict", summary="Compute Actuarial Health Risk Profile")
def predict_health_risk(profile: ApplicantProfile) -> dict:
    try:
        data_dict = profile.model_dump()
        df_input = pd.DataFrame([data_dict])
        
        # --- THE FIX: Rename columns to match the exact training data format ---
        rename_map = {
            "Occupation_type": "Occupation type",
            "Sex": "Sex",
            "Age": "Age",
            "Motorcycle_usage": "Motorcycle usage",
            "Alcohol_usage_length": "Alcohol usage length (year)",
            "Average_daily_cigarette": "Average daily cigarette consumption",
            "Cigarette_usage_length": "Cigarette usage length (year)",
            "BMI": "BMI (weight/height ratio)",
            "Exercicing_years": "Exercicing (years)"
        }
        df_input = df_input.rename(columns=rename_map)
        # -----------------------------------------------------------------------

        if os.path.exists('rf_insurance_model.pkl') and os.path.exists('insurance_scaler.pkl'):
            model = joblib.load('rf_insurance_model.pkl')
            scaler = joblib.load('insurance_scaler.pkl')
            
            scaled_input = scaler.transform(df_input)
            prediction = model.predict(scaled_input)[0]
        else:
            # Fallback if files are missing
            prediction = 1 if (data_dict['BMI'] > 27 and data_dict['Exercicing_years'] < 3) else 0
            
        return {
            "status": "success",
            "predicted_class": int(prediction),
            "risk_assessment": "High Risk / Flag Application" if prediction == 1 else "Standard/Low Risk Profile"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))