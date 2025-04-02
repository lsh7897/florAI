from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from app.recommend import get_flower_recommendations
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.templating import Jinja2Templates

app = FastAPI()

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")

class QueryInput(BaseModel):
    query: Union[str, list[str]]

@app.post("/recommend")
def recommend_flowers(input: QueryInput):
    try:
        query = input.query if isinstance(input.query, str) else ", ".join(input.query)
        return get_flower_recommendations(query)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}
    
    
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})