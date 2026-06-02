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

        reasons = []

    if risk_result == "O":

        if data.glucose < 70:
            reasons.append("혈당이 낮습니다.")

        if data.fasting_hours >= 8:
            reasons.append("공복 시간이 깁니다.")

        if data.insulin_units >= 10:
            reasons.append("인슐린 용량이 높습니다.")

        if data.weight < 55:
            reasons.append("체중이 낮습니다.")

        if len(reasons) == 0:
            reasons.append("모델이 과거 데이터 패턴에서 위험 가능성을 감지했습니다.")

    else:
        reasons.append("현재 입력값에서는 뚜렷한 위험 요인이 감지되지 않았습니다.")
    return {
        "risk_probability": float(prediction),
        "risk_percent": round(float(prediction) * 100, 2),
        "risk_result": risk_result,
        "message": message,
        "reasons": reasons
    }