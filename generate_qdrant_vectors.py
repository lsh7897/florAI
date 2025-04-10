
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_openai import OpenAIEmbeddings

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "flowers"

embedder = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# FLW_IDX 포함된 CSV로 설명 불러오기
csv_data = pd.read_csv("flowers_with_gpt.csv", encoding="utf-8")
csv_data = csv_data.set_index("FLW_IDX")

# JSON 데이터 불러오기
with open("flower_metadata.json", encoding="utf-8") as f:
    flower_data = json.load(f)

# Qdrant 컬렉션 생성
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "desc": VectorParams(size=1536, distance=Distance.COSINE),
        "emotion": VectorParams(size=1536, distance=Distance.COSINE),
        "meaning": VectorParams(size=1536, distance=Distance.COSINE),
    }
)

points = []

for item in tqdm(flower_data):
    try:
        flw_idx = int(item["FLW_IDX"])
        name = item["name"]

        # 🔹 CSV에서 확장된 desc 설명 가져오기
        if flw_idx not in csv_data.index:
            print(f"FLW_IDX {flw_idx} 누락 → CSV 설명 없음, 스킵됨")
            continue
        desc_text = csv_data.loc[flw_idx]["꽃말(설명)"]

        # 🔹 JSON에서 감정, 짧은 꽃말 가져오기
        emo_text = ", ".join(item.get("emotion_tags", []))
        meaning_text = item.get("description", "")
        if not meaning_text:
            print(f"{name} → 짧은 꽃말 없음, 스킵됨")
            continue

        desc_vec = embedder.embed_query(desc_text)
        emo_vec = embedder.embed_query(emo_text)
        meaning_vec = embedder.embed_query(meaning_text)

        points.append(
            PointStruct(
                id=flw_idx,
                payload=item,
                vector={
                    "desc": desc_vec,
                    "emotion": emo_vec,
                    "meaning": meaning_vec
                }
            )
        )
    except Exception as e:
        print(f"{item.get('name', 'Unknown')} 처리 중 오류 발생:", e)

# Qdrant 업로드
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print("Qdrant 벡터 업로드 완료!")
