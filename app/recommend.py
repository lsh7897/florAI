
import os
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# Qdrant 클라이언트
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
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

# 문장 생성: 스타일 대신 꽃말 강조로 변경
def expand_query_components(keywords: list[str]) -> tuple[str, str, str]:
    base = (keywords + [""] * 5)[:5]
    target, emotion, detail, personality, gender = base

    desc = f"{target}에게 {emotion}({detail})의 감정을 전하고 싶은 상황입니다. 상대는 {gender}이고 {personality}인 사람이에요. 이 감정을 말로 대신하지 못하니, 꽃으로 표현하려고 해요."

    emo = f"{target}에게 전하고 싶은 감정은 '{emotion}({detail})'입니다. 말로 표현하기 어려운 감정을 꽃을 통해 대신 전하고 싶습니다."

    meaning = f"{emotion}({detail})의 감정을 가장 잘 상징할 수 있는 꽃말을 가진 꽃을 추천해주세요. 이 감정의 의미를 강하게 전달하는 상징적인 꽃이 필요해요."

    return desc, emo, meaning

# GPT 설명 생성
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

# 추천
def get_flower_recommendations(keywords: list[str], top_k: int = 3):
    desc_query, emo_query, meaning_query = expand_query_components(keywords)

    desc_vec = embedder.embed_query(desc_query)
    emo_vec = embedder.embed_query(emo_query)
    meaning_vec = embedder.embed_query(meaning_query)

    desc_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector={"name": "desc", "vector": desc_vec},
        limit=15
    )
    emo_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector={"name": "emotion", "vector": emo_vec},
        limit=15
    )
    meaning_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector={"name": "meaning", "vector": meaning_vec},
        limit=15
    )

    def normalize(scores):
        arr = np.array(scores)
        min_v, max_v = arr.min(), arr.max()
        return (arr - min_v) / (max_v - min_v + 1e-8)

    def build_score_map(results):
        return {r.payload["name"]: r.score for r in results}

    desc_map = build_score_map(desc_results)
    emo_map = build_score_map(emo_results)
    meaning_map = build_score_map(meaning_results)

    all_names = set(desc_map) | set(emo_map) | set(meaning_map)

    desc_norm = normalize([desc_map.get(n, 0) for n in all_names])
    emo_norm = normalize([emo_map.get(n, 0) for n in all_names])
    meaning_norm = normalize([meaning_map.get(n, 0) for n in all_names])

    weights = {"desc": 0.4, "emotion": 0.3, "meaning": 0.3}
    final_scores = {}
    for i, name in enumerate(all_names):
        final_scores[name] = (
            desc_norm[i] * weights["desc"] +
            emo_norm[i] * weights["emotion"] +
            meaning_norm[i] * weights["meaning"]
        )

    sorted_names = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_names[:top_k]

    final = []
    seen = set()
    for name, score in selected:
        for r in desc_results + emo_results + meaning_results:
            if r.payload["name"] == name and name not in seen:
                seen.add(name)
                reason = generate_reason(desc_query, r.payload["description"], name)
                final.append({
                    "FLW_IDX": r.payload["FLW_IDX"],
                    "name": name,
                    "score": round(score, 4),
                    "reason": reason
                })
                break

    return {"recommendations": final}
