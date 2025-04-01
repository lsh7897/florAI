from fastapi import FastAPI
from pydantic import BaseModel
from app.recommend import get_flower_recommendations

# 여기 반드시 있어야 함!!!
app = FastAPI()

# Pydantic 모델 정의
class QueryInput(BaseModel):
    query: str

# 엔드포인트
@app.post("/recommend")
def recommend_flowers(input: QueryInput):
    return get_flower_recommendations(input.query)

# 기본 root 테스트용 엔드포인트
@app.get("/")
def root():
    return {"message": "FlorAI 꽃 추천 API 정상 작동 중 🌸"}
