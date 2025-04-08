
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 환경변수
QDRANT_URL = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "flowers"

# 클라이언트 초기화
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# 임베딩 모델
embedder = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

# JSON 파일 로딩
with open("flower_metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 컬렉션 재생성
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "desc": rest.VectorParams(size=1536, distance=rest.Distance.COSINE),
        "emotion": rest.VectorParams(size=1536, distance=rest.Distance.COSINE),
        "style": rest.VectorParams(size=1536, distance=rest.Distance.COSINE),
    }
)

# 벡터 및 페이로드 구성
points = []
for idx, item in enumerate(tqdm(data)):
    name = item["name"]
    desc = item["description"]
    emotions = item.get("emotion_tags", [])
    flw_idx = item["FLW_IDX"]

    # 문장 생성
    desc_text = desc
    emo_text = f"이 꽃은 감정적으로 {', '.join(emotions)}와 관련이 있습니다." if emotions else "감정 정보 없음"
    style_text = f"{item['color']} 색이며 {item['smell']} 향기를 가지고 있습니다. 계절: " + ", ".join(
        [s for s in ["봄", "여름", "가을", "겨울"] if item.get(f"season_{s}", False)]
    )

    # 임베딩
    try:
        desc_vec = embedder.embed_query(desc_text)
        emo_vec = embedder.embed_query(emo_text)
        style_vec = embedder.embed_query(style_text)
    except Exception as e:
        print(f"❌ 임베딩 실패: {name} - {e}")
        continue

    payload = {
        "name": name,
        "description": desc,
        "emotion_tags": emotions,
        "FLW_IDX": flw_idx
    }

    points.append(rest.PointStruct(
        id=idx,
        vector={"desc": desc_vec, "emotion": emo_vec, "style": style_vec},
        payload=payload
    ))

# 업로드
qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"✅ 업로드 완료: {len(points)}개 꽃 정보 Qdrant에 저장됨")
