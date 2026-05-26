import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# CSV 읽기
df = pd.read_csv("data.csv")

# 입력 데이터
X = df[[
    "glucose",
    "fasting_hours",
    "insulin_units",
    "weight"
]]

# 정답 데이터
y = df["result"]

# 학습용/테스트용 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 모델 생성
model = XGBClassifier()

# 모델 학습
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# 모델 저장
joblib.dump(model, "model.pkl")

print("모델 저장 완료")
