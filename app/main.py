from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Union
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import faiss
import json
import os
import numpy as np
from app.utils import embed_query, generate_reason
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.recommend import get_flower_recommendations

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class QueryInput(BaseModel):
    query: Union[str, list[str]]

@app.post("/recommend")
def recommend_flowers(input: QueryInput):
    try:
        query = input.query if isinstance(input.query, str) else ", ".join(input.query)
        result = get_flower_recommendations(query)
        return result["recommendations"]
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "FlorAI 꽃 추천 API",
        "heading": "FlorAI 꽃 추천 API",
        "description": "이 페이지는 FlorAI API 서버가 정상 작동 중임을 나타내며, 예시적인 패턴을 보여줍니다.",
        "example_query": "사랑 고밝"
    })