from fastapi import FastAPI
from pydantic import BaseModel
from app.recommend import get_flower_recommendations  # 여기에 추천 로직 분리

app = FastAPI()

# 입력 모델 정의
class QueryInput(BaseModel):
    query: str

@app.post("/recommend")
def recommend(input: QueryInput):
    try:
        # 핵심 추천 함수 호출
        result = get_flower_recommendations(input.query)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "FlorAI 꽃 추천 API 정상 작동 중 🌸"}