
import os
import json
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Qdrant client setup
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Embedding model setup
embedder = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

# Load data
df = pd.read_csv("flowers_with_gpt.csv")
with open("flower_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

# Convert metadata to dict by FLW_IDX
metadata_dict = {item["FLW_IDX"]: item for item in metadata}

# Vector field names
VECTOR_DIM = 1536
COLLECTION_NAME = "flowers"

# Create Qdrant collection
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "desc": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        "emotion": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        "style": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    }
)

# Prepare and insert points
points = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    flw_idx = row["FLW_IDX"]
    flower_meta = metadata_dict.get(flw_idx, {})

    # Description vector
    desc_text = str(row["꽃말(설명)"])
    desc_vec = embedder.embed_query(desc_text)

    # Emotion vector
    emotion_tags = flower_meta.get("emotion_tags", [])
    emotion_sentence = "이 꽃은 " + ", ".join(emotion_tags) + "의 감정을 담고 있어요."
    emotion_vec = embedder.embed_query(emotion_sentence)

    # Style vector
    color = flower_meta.get("color", "")
    scent = flower_meta.get("scent", "")
    seasons = [k for k in ["spring", "summer", "autumn", "winter"] if flower_meta.get(k)]
    season_text = ", ".join(seasons)
    style_sentence = f"{color} 색의 꽃이며, 향기는 {scent}이고, 계절은 {season_text}입니다."
    style_vec = embedder.embed_query(style_sentence)

    # Payload
    payload = {
        "FLW_IDX": flw_idx,
        "name": row["꽃 이름"],
        "description": desc_text,
        "emotion_tags": emotion_tags,
        "color": color,
        "scent": scent,
        "season": seasons
    }

    points.append(PointStruct(
        id=int(flw_idx.replace("F", "")),  # Qdrant ID는 int로
        vector={
            "desc": desc_vec,
            "emotion": emotion_vec,
            "style": style_vec
        },
        payload=payload
    ))

# Upload points in batch
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
