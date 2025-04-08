
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Union
import traceback
from app.recommend import get_flower_recommendations

app = FastAPI()

class QueryInput(BaseModel):
    query: Union[str, list[str]]

@app.post("/recommend")
def recommend_flowers(input: QueryInput):
    try:
        query = input.query if isinstance(input.query, list) else [k.strip() for k in input.query.split(",")]
        result = get_flower_recommendations(query)
        return result["recommendations"]
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

@app.get("/")
async def index():
    return {
        "message": "FlorAI 꽃 추천 API가 정상 작동 중입니다.",
        "example_query": "자녀, 슬픔, 그리움, 차분한 사람입니다"
    }
