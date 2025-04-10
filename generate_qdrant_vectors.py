import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_openai import OpenAIEmbeddings

# 🔹 환경변수 로딩
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 🔹 Qdrant 클라이언트 생성
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 🔹 컬렉션 이름
COLLECTION_NAME = "flowers"

# 🔹 OpenAI 임베딩 모델 초기화
embedder = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

# 🔹 벡터 정규화 함수 추가
def normalize(v):
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm != 0 else v

# 🔹 CSV 로드 (FLW_IDX 기준)
csv_data = pd.read_csv("flowers_with_gpt.csv", encoding="utf-8")
csv_data = csv_data.set_index("FLW_IDX")

# 🔹 JSON 데이터 로드
with open("flower_metadata.json", encoding="utf-8") as f:
    flower_data = json.load(f)

# 🔹 Qdrant 컬렉션 재생성
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "desc": VectorParams(size=1536, distance=Distance.COSINE),
        "emotion": VectorParams(size=1536, distance=Distance.COSINE),
        "meaning": VectorParams(size=1536, distance=Distance.COSINE),
    }
)

points = []

# 🔹 벡터 생성 및 수집
for item in tqdm(flower_data):
    try:
        flw_idx = int(item["FLW_IDX"])
        name = item["name"]

        if flw_idx not in csv_data.index:
            print(f"FLW_IDX {flw_idx} 누락 → CSV 설명 없음, 스킵됨")
            continue
        desc_text = csv_data.loc[flw_idx]["꽃말(설명)"]

        emo_text = ", ".join(item.get("emotion_tags", []))
        meaning_text = item.get("description", "")
        if not meaning_text:
            print(f"{name} → 짧은 꽃말 없음, 스킵됨")
            continue

        desc_vec = normalize(embedder.embed_query(desc_text))
        emo_vec = normalize(embedder.embed_query(emo_text))
        meaning_vec = normalize(embedder.embed_query(meaning_text))

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

# 🔹 Qdrant 업로드
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print("✅ Qdrant 벡터 업로드 완료!")