from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    input_data = pd.DataFrame([{
        "glucose": data.glucose,
        "fasting_hours": data.fasting_hours,
        "insulin_units": data.insulin_units,
        "weight": data.weight
    }])

    # AI 모델 예측
    ai_probability = model.predict_proba(input_data)[0][1]

    # 규칙 기반 위험 점수
    risk_score = 0
    reasons = []

    # 혈당
    if data.glucose < 70:
        risk_score += 3
        reasons.append("혈당이 70mg/dL 미만으로 낮아 저혈당 위험이 높습니다.")
    elif data.glucose < 90:
        risk_score += 1
        reasons.append("혈당이 정상보다 낮은 편입니다.")

    # 공복 시간
    if data.fasting_hours >= 8:
        risk_score += 2
        reasons.append("공복 시간이 길어 혈당이 더 떨어질 수 있습니다.")
    elif data.fasting_hours >= 4:
        risk_score += 1
        reasons.append("공복 시간이 다소 긴 편입니다.")

    # 인슐린 용량
    if data.insulin_units >= 10:
        risk_score += 2
        reasons.append("인슐린 용량이 높아 혈당 하강 위험이 있습니다.")
    elif data.insulin_units >= 5:
        risk_score += 1
        reasons.append("인슐린 용량이 중간 이상입니다.")

    # 체중
    if data.weight < 55:
        risk_score += 1
        reasons.append("체중이 낮아 인슐린 영향이 더 크게 나타날 수 있습니다.")

    # 최종 판정
    if risk_score >= 3:
        risk_result = "O"
        message = "위험"
    else:
        risk_result = "X"
        message = "안 위험"

    if len(reasons) == 0:
        reasons.append("입력값에서 뚜렷한 위험 요인이 감지되지 않았습니다.")

    return {
        "ai_probability": float(ai_probability),
        "ai_percent": round(float(ai_probability) * 100, 2),
        "risk_score": risk_score,
        "risk_result": risk_result,
        "message": message,
        "reasons": reasons
    }