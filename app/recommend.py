
import os
import numpy as np
import random
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# Qdrant 설정
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)
COLLECTION_NAME = "flowers"

# 임베딩 & LLM
embedder = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

# 문장 생성
def expand_query_desc(keywords: list[str]) -> tuple[str, list[str], list[str]]:
    base = (keywords + [""] * 5)[:5]
    target, emotion, detail, personality, gender = base

    desc = f"{target}에게 {emotion}({detail})의 감정을 표현하고 싶어요. 상대는 {gender}이고 {personality}입니다."
    emotion_tags = [f"{emotion}({detail})"]
    style_keywords = [gender, personality]
    return desc, emotion_tags, style_keywords

# 추천 이유 생성
def generate_reason(query: str, description: str, flower_name: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query", "description", "flower"],
        template="""
        사용자 의도: {query}
        꽃 설명: {description}
        이 꽃이 '{query}'에 어울리는 이유를 두 문장이상으로 설명해줘. 꽃 이름({flower})도 포함해서 구매자를 충분히 설득 할 수 있도록 구체적으로 설명해줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "query": query,
        "description": description,
        "flower": flower_name
    }).strip()

# 추천 함수
def get_flower_recommendations(keywords: list[str], top_k: int = 3, candidate_k: int = 15):
    desc_query, emotion_tags, style_keywords = expand_query_desc(keywords)
    desc_vec = embedder.embed_query(desc_query)

    print("📌 쿼리 문장:", desc_query)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector={"name": "desc", "vector": desc_vec},
        limit=candidate_k
    )

    print("\n📊 후보 유사도 점수:")
    for r in results[:10]:
        print(f"  - {r.payload['name']}: {r.score:.4f}")

    scored = []
    for r in results:
        payload = r.payload
        score = r.score
        boost = 0.0

        # 감정 태그 가중치 (정확히 일치하는 태그가 있는 경우)
        if any(tag in payload.get("emotion_tags", []) for tag in emotion_tags):
            boost += 0.05

        # 스타일 키워드 (성향/성별/색/향기 등에서 일치 단어 존재시)
        if any(sk in payload.get("description", "") for sk in style_keywords):
            boost += 0.03

        scored.append((r, score + boost))

    # 샘플링 기반 다양성 + 최종 점수 정렬
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    sampled = scored[:top_k * 2] if len(scored) >= top_k * 2 else scored

    final = []
    seen = set()
    for r, s in sampled:
        name = r.payload["name"]
        if name in seen:
            continue
        seen.add(name)

        reason = generate_reason(desc_query, r.payload["description"], name)
        final.append({
            "FLW_IDX": r.payload["FLW_IDX"],
            "name": name,
            "score": round(s, 4),
            "reason": reason
        })

        if len(final) >= top_k:
            break

    return {"recommendations": final}
