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
    try:
        return get_flower_recommendations(input.query)
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # 서버 콘솔에 전체 에러 출력
        return {"error": str(e)}       # 클라이언트에도 에러 메시지 전달

# 기본 root 테스트용 엔드포인트
@app.get("/")
def root():
    return {"message": "FlorAI 꽃 추천 API 정상 작동 중 🌸"}
