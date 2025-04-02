from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.recommend import get_flower_recommendations

app = FastAPI()

class QueryInput(BaseModel):
    query: Union[str, list[str]]  # ✅ 문자열 or 리스트 둘 다 허용

@app.post("/recommend")
def recommend_flowers(input: QueryInput):
    try:
        # 만약 리스트로 들어오면 쉼표로 묶어서 문자열로 만들기
        query = input.query if isinstance(input.query, str) else ", ".join(input.query)
        return get_flower_recommendations(query)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "FlorAI 꽃 추천 API 정상 작동 중 🌸"}