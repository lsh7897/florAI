
import numpy as np
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()

# 임베딩 초기화
embedder = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

def embed_text(text: str) -> list[float]:
    """문장을 OpenAI 임베딩 벡터로 변환"""
    return embedder.embed_query(text)

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 벡터의 코사인 유사도 계산"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
