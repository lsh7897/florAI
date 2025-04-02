from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from app.recommend import get_flower_recommendations

app = FastAPI()

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