from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# 모델 불러오기
model = joblib.load("model.pkl")

# 입력 형식
class PatientData(BaseModel):
    glucose: int
    fasting_hours: int
    insulin_units: int
    weight: int

@app.post("/predict")
def predict(data: PatientData):

    # DataFrame 형태로 변환
    input_data = pd.DataFrame([{
        "glucose": data.glucose,
        "fasting_hours": data.fasting_hours,
        "insulin_units": data.insulin_units,
        "weight": data.weight
    }])

    # 예측
    prediction = model.predict_proba(input_data)[0][1]

    # 위험 여부 판단
    if prediction >= 0.5:
        risk_result = "O"
        message = "위험"
    else:
        risk_result = "X"
        message = "안 위험"

    return {
        "risk_probability": float(prediction),
        "risk_result": risk_result,
        "message": message
    }
