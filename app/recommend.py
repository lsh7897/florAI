
import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

# Qdrant 연결
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "flowers"

# 임베딩 & GPT
embedder = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-ada-002")
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# 감정 프롬프트 구성
def expand_query_components(keywords: list[str]):
    if len(keywords) < 5:
        keywords += [""] * (5 - len(keywords))
    target, emotion, detail, personality, gender = keywords
    desc = f"{target}에게 {emotion}({detail})의 감정을 표현하고 싶어요. 상대는 {gender}이고 {personality}입니다."
    emo = f"이 감정은 {emotion}({detail})입니다."
    style = f"{gender}이고 {personality} 성향의 사람에게 어울릴만한 꽃을 추천해줘."
    return desc, emo, style

# GPT 설명 생성
def generate_reason(query: str, description: str, flower_name: str, flower_meaning: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query", "description", "flower", "meaning"],
        template="""
        사용자 의도: {query}
        꽃 설명: {description}
        꽃말: {meaning}

        이 꽃이 '{query}'에 어울리는 이유를 구체적으로 설명해줘.
        꽃 이름({flower})도 반드시 포함하고, 꽃말을 중심으로 감정과 메시지를 설득력 있게 표현해줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "query": query,
        "description": description,
        "flower": flower_name,
        "meaning": flower_meaning
    }).strip()

# 추천 메인 함수
def get_flower_recommendations(keywords: list[str], top_k: int = 3):
    desc_query, emo_query, style_query = expand_query_components(keywords)

    # 벡터 생성
    desc_vec = embedder.embed_query(desc_query)
    emo_vec = embedder.embed_query(emo_query)
    meaning_vec = embedder.embed_query(style_query)

    # 검색
    SEARCH_TOP_K = 30
    vectors = {"desc": desc_vec, "emotion": emo_vec, "meaning": meaning_vec}
    results = {
        name: qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector={"name": name, "vector": vector},
            limit=SEARCH_TOP_K
        )
        for name, vector in vectors.items()
    }

    # 가중치 설정
    weights = {"desc": 0.6, "emotion": 0.25, "meaning": 0.15}
    score_map = {}

    for vector_name, result in results.items():
        for res in result:
            name = res.payload["name"]
            score = res.score
            score_map.setdefault(name, []).append((vector_name, score))

    # 최종 점수 계산
    flower_scores = []
    for name, scores in score_map.items():
        score_total = 0.0
        used = {"desc": 0.0, "emotion": 0.0, "meaning": 0.0}
        for vector_name, score in scores:
            used[vector_name] = score
        for k, v in weights.items():
            score_total += used[k] * v
        flower_scores.append((name, score_total))

    flower_scores.sort(key=lambda x: x[1], reverse=True)
    top_names = [x[0] for x in flower_scores[:top_k]]

    # 최종 추천 정리
    final_recommendations = []
    seen = set()
    for vector_name, result in results.items():
        for res in result:
            payload = res.payload
            if payload["name"] in top_names and payload["name"] not in seen:
                seen.add(payload["name"])
                reason = generate_reason(
                    query=",".join(keywords),
                    description=payload["description"],
                    flower_name=payload["name"],
                    flower_meaning=payload.get("description", "")
                )
                final_recommendations.append({
                    "FLW_IDX": payload["FLW_IDX"],
                    "name": payload["name"],
                    "score": round(score_map[payload["name"]][0][1], 4),
                    "reason": reason
                })
                break

    return {"recommendations": final_recommendations}
