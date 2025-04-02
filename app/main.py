from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.recommend import get_flower_recommendations

app = FastAPI()

class QueryInput(BaseModel):
    query: List[str]  # ✅ 리스트로 받는다!

@app.post("/recommend")
def recommend(input: QueryInput):
    return get_flower_recommendations(input.query)


@app.get("/")
def root():
    return {"message": "FlorAI 꽃 추천 API 정상 작동 중 🌸"}