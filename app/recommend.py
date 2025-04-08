
import os
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.utils import cosine_similarity, embed_text

load_dotenv()

# Qdrant 클라이언트 - HTTP API 모드로 설정
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "flowers"

# OpenAI 임베딩
embedder = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

# GPT 모델
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

def expand_query_components(keywords: list[str]) -> tuple[str, str, str]:
    base = (keywords + [""] * 5)[:5]
    target, main_emotion, detail_emotion, personality, gender = base

    desc = (
        f"{main_emotion}이 제일 핵심적인 키워드로서 "
        f"{target}에게 {main_emotion}에 대한 감정을 표현하고 싶어. "
        f"{detail_emotion} {main_emotion}을 생각하며 꽃을 받는 상대방은 {personality}."
    )
    emo = f"이 감정은 {main_emotion}({detail_emotion})입니다. 대상은 {gender}입니다."
    style = (
        f"{gender}이고 {personality} 성향의 사람에게 어울릴만한 "
        f"색, 향기, 계절감을 가진 꽃을 추천해줘."
    )
    return desc, emo, style

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

def get_flower_recommendations(keywords: list[str], top_k: int = 3):
    desc_query, emo_query, style_query = expand_query_components(keywords)

    desc_vec = embedder.embed_query(desc_query)
    emo_vec = embedder.embed_query(emo_query)
    style_vec = embedder.embed_query(style_query)

    # search_collection 사용
    results = {
        "desc": qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector={"name": "desc", "vector": desc_vec},
            limit=top_k * 5
        ),
        "emotion": qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector={"name": "emotion", "vector": emo_vec},
            limit=top_k * 5
        ),
        "style": qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector={"name": "style", "vector": style_vec},
            limit=top_k * 5
        )
    }

    # ID 통합 및 점수 계산
    candidate_ids = set(hit.id for hits in results.values() for hit in hits)
    scored = []
    for idx in candidate_ids:
        d_score = next((r.score for r in results["desc"] if r.id == idx), 0.0)
        e_score = next((r.score for r in results["emotion"] if r.id == idx), 0.0)
        s_score = next((r.score for r in results["style"] if r.id == idx), 0.0)
        final_score = 0.5 * d_score + 0.3 * e_score + 0.2 * s_score

        payload = next((r.payload for r in results["desc"] + results["emotion"] + results["style"] if r.id == idx), {})
        scored.append((payload, final_score))

    # 정렬 및 결과 생성
    scored.sort(key=lambda x: x[1], reverse=True)
    final = []
    for payload, score in scored:
        reason = generate_reason(desc_query, payload["description"], payload["name"])
        final.append({
            "FLW_IDX": payload["FLW_IDX"],
            "name": payload["name"],
            "score": round(score, 4),
            "reason": reason
        })
        if len(final) >= top_k:
            break

    return {"recommendations": final}
