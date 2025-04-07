from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Union
import traceback
from app.recommend import get_flower_recommendations

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class QueryInput(BaseModel):
    query: Union[str, list[str]]

@app.post("/recommend")
def recommend_flowers(input: QueryInput):
    try:
        # 입력값이 리스트일 경우 그대로, 아니면 ","로 구분하여 리스트로 변경
        query = input.query if isinstance(input.query, list) else input.query.split(",")
        
        # 꽃 추천 함수 호출
        result = get_flower_recommendations(query)
        
        # 빈 값이 반환된 경우에 대해 체크
        if not result.get("recommendations"):
            return {"error": "추천할 꽃이 없습니다.", "recommendations": []}

        return result  # 정상적인 추천 결과 반환

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "FlorAI 꽃 추천 API",
        "heading": "FlorAI 꽃 추천 API",
        "description": "이 페이지는 FlorAI API 서버가 정상 작동 중임을 나타내며, 예시적인 패턴을 보여줍니다.",
        "example_query": "부모, 슬픔, 그리움, 차분한 사람입니다"
    })
