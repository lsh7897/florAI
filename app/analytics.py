from fastapi import APIRouter
import pandas as pd
import os

router = APIRouter()

@router.get("/analytics")
def keyword_chart_data():
    path = "logs/keyword_log.csv"
    if not os.path.exists(path):
        return {"error": "로그 파일이 없습니다."}

    df = pd.read_csv(path)
    stats = {}
    for col in df.columns[1:]:  # 첫 컬럼은 timestamp
        stats[col] = df[col].value_counts().to_dict()

    return stats